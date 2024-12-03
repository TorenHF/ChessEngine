import numpy as np
import torch
import torch.nn.functional as F
import random
import chess
import torch.multiprocessing as mp
from functools import partial

from Chess import selfPlay_wrapper


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, node, mcts):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = mcts
        self.state = chess.Board()
        self.root = None

    def selfPlay(self):
        return_memory = []
        memory = []
        player = 1
        state = chess.Board()
        num_moves = 0
        is_terminal = False

        print("s")
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
                #print(num_moves)
                for hist_neutral_state, hist_action_probs, hist_player, hist_q_value in memory:
                    hist_outcome = value if hist_player == player else self.game.getOpponentValue(value)
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state).cpu(),
                        hist_action_probs,
                        hist_outcome,  # Use the final outcome as z
                        hist_q_value  # Save q for training
                    ))



            player = self.game.getOpponent(player)

        return return_memory

    # Here there is a possibility to add q in training
    def train(self, memory):

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(device)
        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) -1, batchIdx+self.args['batch_size'])]

            state, policy_targets, value_targets, q_values = zip(*sample)


            q_values = np.array(q_values).reshape(-1, 1)  # Reshape q_values

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            avg_qz = (q_values + value_targets) / 2.0

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

            with open('output.loss_data', 'a') as f:
                f.write('\n' + str(loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.model.to('cpu')

    def learn(self):
        # Move the model to CPU to make it picklable
        mp.set_start_method('spawn', force=True)
        # Move the model to CPU memory so it can be shared (pickled) with worker processes
        self.model.to('cpu')

        max_games = self.args['num_max_parallel_batches'] * self.args['num_parallel_games'] + self.args['num_parallel_games']
        start_parameter_mcts = self.args['num_searches'] - self.args['num_max_searches']
        start_parameter_spg = self.args['num_parallel_games'] - max_games

        with mp.Pool(processes=self.args['num_parallel_games']) as pool:
            for iteration in range(self.args['num_iterations']):
                self.model.eval()
                memory = []

                self.args['num_searches'] =  round(float(start_parameter_mcts * np.exp(-iteration * 0.5) + self.args['num_max_searches']))
                self.args['num_selfPlay_iterations'] = round(float(start_parameter_spg * np.exp(-iteration * 0.5) + max_games))


                selfPlay_partial = partial(selfPlay_wrapper, self.mcts, self.game, self.args, self.model,
                                           self.model.state_dict())


                num_batches = self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']



                for selfPlay_iteration in range(num_batches):
                    async_results = []
                    # Distribute self-play tasks to worker processes using the partial function
                    results = pool.map(selfPlay_partial, range(self.args['num_parallel_games']))

                    # Combine all game memories
                    for idx, (success, result, num_moves) in enumerate(results):
                        if success:
                            game_memory = result
                            memory.extend(game_memory)
                            with open('output.num_moves.data', 'a') as f:
                                f.write('\n' + str(num_moves))



                # Training phase
                self.model.train()
                for epoch in range(self.args['num_epochs']):
                    self.train(memory)

                # After training, move the model back to CPU for pickling in next iteration
                self.model.to('cpu')

                # Save model and optimizer state
                torch.save(self.model.state_dict(), f"model_{iteration}.pt")
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")












    def learn_old(self):
            mp.set_start_method('spawn', force=True)
            for iteration in range(self.args['num_iterations']):
                memory = []


                self.model.eval()
                for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                    queue = mp.Queue()
                    processes = []

                    for i in range(self.args['num_parallel_games']):
                        p = mp.Process(target=self.selfPlay_wrapper, args=(queue, i))
                        processes.append(p)
                        p.start()

                    # Gather results
                    game_memory = [queue.get() for _ in range(self.args['num_parallel_games'])]
                    memory.extend(game_memory)

                    for p in processes:
                        p.join()

                    print(f"iteration: {iteration}, game: {selfPlay_iteration}")
                print("complete")


                self.model.train()
                for epoch in range(self.args['num_epochs']):
                    self.train(memory)

                torch.save(self.model.state_dict(), f"model_{iteration}.pt" )
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}.pt")

    def selfPlay_wrapper_old(self, model_state_dict, i):
        # Load the model state dictionary for this worker
        self.model.load_state_dict(model_state_dict)
        # Move model to the device (GPU or CPU) as needed
        self.model.to("mps" if torch.backends.mps.is_available() else "cpu")

        # Perform a self-play game and return its memory
        return self.selfPlay()

    def selfPlay_wrapper_1(self, model_state_dict, idx):
        try:
            # Initialize device
            # Load the model state dictionary for this worker
            self.model.load_state_dict(model_state_dict)

            # Move model to the device (GPU or CPU) as needed
            self.model.to("mps" if torch.backends.mps.is_available() else "cpu")
            self.model.eval()


            result = self.selfPlay()

            return (True, result)  # Return the result to the main process

        except Exception as e:
            print(f"Worker {idx}: Exception occurred: {e}")
            return (False, e)# Re-raise exception to be caught in the main process


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


