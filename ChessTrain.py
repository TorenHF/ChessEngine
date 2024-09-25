import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import Chess
import chess



class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args['num_parallel_games'])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])

            neutral_states = self.game.changePerspective(states, player)
            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.actionSize)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)  # Ensure it sums to 1
                action = np.random.choice(len(states.legal_moves), p=temperature_action_probs)
                move = self.game.actionToMove(states, action, player)

                spg.state = self.game.makeMoveAZ(spg.state, action)

                value, is_terminal = self.game.get_value_and_terminate(spg.state, action)
                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = value if hist_player == player else self.game.getOpponentValue(value)
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del spGames[i]

            player = self.game.getOpponent(player)

        return return_memory



    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) -1, batchIdx+self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            state = torch.tensor(state,dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def learn(self):
        for iteration in range(self.args['num_iterations']):
            memory = []


            self.model.eval()
            for selfPlay_iteration in range(self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']):
                memory += self.selfPlay()
                print(f"iteration: {iteration}, game: {selfPlay_iteration}")


            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train(memory)

            torch.save(self.model.state_dict(), f"model_{iteration}_{self.game}.pt" )
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_{self.game}.pt")

class SPG:
    def __init__(self, game):
        self.state = chess.Board()
        self.memory = []
        self.root = None
        self.node = None
class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, axis=1).cpu().numpy()
        policy = ((1-self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']]* self.game.actionSize, size=policy.shape[0]))

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = spg.state.legal_moves
            #binary valid moves
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Chess.Node(self.game, self.args, spg.state, visit_count=1)

            spg.root.expand(spg_policy)

        for search in range(self.args['num_searches']):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
                value = self.game.getOpponentValue(value)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

                expandable_spGames = [mappingIdx for mappingIdx in range(len(spGames)) if spGames[mappingIdx].node is not None]

                if len(expandable_spGames) >0:
                    states = np.stack([spGames[mappingIdx].node.state for mappingIdx in expandable_spGames])

                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                    )
                    policy = torch.softmax(policy, axis=1).cpu().numpy()
                    value = value.cpu().numpy()

                for i, mappingIdx in enumerate(expandable_spGames):
                    spg_policy, spg_value = policy[i], value[i]
                    node = spGames[mappingIdx].node

                    valid_moves = self.game.getValidMoves(node.state)
                    #binary valid moves
                    spg_policy *= valid_moves
                    spg_policy /= np.sum(spg_policy)

                    node.expand(spg_policy)
                    node.backpropagate(spg_value)


