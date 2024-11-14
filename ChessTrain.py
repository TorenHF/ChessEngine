import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import chess
import torch.multiprocessing as mp




class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, node, mcts):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = mcts
        self.root = None

    def selfPlay(self):
        return_memory = []
        memory = []
        player = 1
        state = chess.Board()
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]
        num_moves = 0
        is_terminal = False
        print("start")

        while not is_terminal:
            num_moves += 1

            neutral_state = self.game.changePerspective(state, player)
            action_probs, root = self.mcts.search(neutral_state)


            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)  # Ensure it sums to 1

            action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)
            move = self.game.all_moves[action]
            move = self.game.flip_move(move, player)
            state.push(move)

            value, is_terminal = self.game.get_value_and_terminate(state, num_moves)

            # Save the π (action probabilities) and the Q value of the root node
            q_value = root.value_sum / root.visit_count  # Q-value for the root node
            memory.append((neutral_state, action_probs, player, q_value))  # Store q_value

            if is_terminal:
                print("done")
                for hist_neutral_state, hist_action_probs, hist_player, hist_q_value in memory:
                    hist_outcome = value if hist_player == player else self.game.getOpponentValue(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome,  # Use the final outcome as z
                        hist_q_value  # Save q for training
                    ))



            player = self.game.getOpponent(player)

        return return_memory

    # Here there is a possibility to add q in training
    def train(self, memory):

        for game in range(2):
            game_memory = memory[game]
            random.shuffle(game_memory)
            for batchIdx in range(0, len(memory), self.args['batch_size']):
                sample = game_memory[batchIdx:min(len(memory) -1, batchIdx+self.args['batch_size'])]

                state, policy_targets, value_targets, q_values = zip(*sample)

                q_values = np.array(q_values).reshape(-1, 1)  # Reshape q_values

                # Compute the average of q and z (value_targets)
                avg_qz = (q_values + value_targets) / 2.0

                state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

                state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
                policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
                avg_qz = torch.tensor(avg_qz, dtype=torch.float32, device=self.model.device)

                out_policy, out_value = self.model(state)

                # Policy loss remains the same
                policy_loss = F.cross_entropy(out_policy, policy_targets)

                # Value loss: train on the average of q and z
                value_loss = F.mse_loss(out_value, avg_qz)

                # Total loss
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []


            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                with mp.Pool(processes=2) as pool:
                    # Run selfPlay for each game_id in parallel
                    memory = pool.map(self.selfPlay_wrapper, range(2))
                print(f"iteration: {iteration}, game: {selfPlay_iteration}")
            print("complete")


            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}.pt" )
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    def selfPlay_wrapper(self, idx):
        return self.selfPlay()

class SPG:
    def __init__(self, game):
        self.state = chess.Board()
        self.game = game
        self.memory = []
        self.root = None
        self.node = None
class MCTSParallel:
    def __init__(self, game, args, model, node):
        self.game = game
        self.args = args
        self.model = model
        self.node = node

    @torch.no_grad()
    def search(self, states, spGames, player):
        policy, _ = self.model(
            self.game.get_encoded_state_parallel(states).clone().detach().to(self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = ((1-self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']]* self.game.actionSize, size=policy.shape[0]))

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_binary_moves(states[i])


            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = self.node(self.game, self.args, spg.state, visit_count=1)
            spg.root.state = states[i]


            spg.root.expand(spg_policy, player)

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminate(node.state, 1)
                value = self.game.getOpponentValue(value)

                if is_terminal:
                    node.backpropagate(value)


                else:
                    spg.node = node


                expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

                if len(expandable_spGames) >0:
                    states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                    policy, value = self.model(
                        self.game.get_encoded_state_parallel(states)
                    )
                    policy = torch.softmax(policy, axis=1).cpu().numpy()
                    value = value.cpu().numpy()

                for i, mappingIdx in enumerate(expandable_spGames):
                    spg_policy, spg_value = policy[i], value[i]
                    node = spGames[mappingIdx].node


                    valid_moves = self.game.get_binary_moves(node.state)

                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)

                    node.expand(spg_policy, player)
                    node.backpropagate(spg_value)


