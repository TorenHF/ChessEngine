import chess
import chess.engine
import Chess
import torch
import torch.nn.functional as F
import numpy as np
import ChessTrain
import random
import math
from stockfish import Stockfish

class StockfishTrain:
    def __init__(self, game, model, stockfish, optimizer, args):
        self.game = game
        self.model = model
        self.optimizer = optimizer
        self.stockfish = stockfish
        self.args = args


    def play_game(self):
        board = chess.Board()
        player = 1
        hMove_counter = 0
        value, is_terminal = self.game.get_value_and_terminate(board, hMove_counter)
        memory = []
        return_memory = []
        while not is_terminal:
            flipped_board = self.game.changePerspective(board, player)
            neutral_state = self.game.get_encoded_state(flipped_board)
            copy_board = board.copy()


            memory.append((neutral_state, copy_board))

            if hMove_counter % 4 == 0 or hMove_counter % 3 == 0:
                legal_moves = self.game.get_binary_moves(flipped_board)
                move_probs = legal_moves / np.sum(legal_moves)
                action = np.random.choice(len(self.game.all_moves), p=move_probs)
                move = self.game.all_moves[action]
            else:
                self.stockfish.set_fen_position(flipped_board.fen())
                move = self.stockfish.get_best_move_time(300)
                move = chess.Move.from_uci(move)

            move = self.game.flip_move(move, player)
            board.push(move)

            hMove_counter += 1
            value, is_terminal = self.game.get_value_and_terminate(board, hMove_counter)
            player = self.game.getOpponent(player)


            if is_terminal:
                for hist_neutral_state, hist_board in memory:
                    eval = self.get_stockfish_eval(hist_board)
                    value_target = self.scale_tanh(eval)
                    move_probs = self.get_stockfish_move_probs(hist_board)
                    return_memory.append((
                        hist_neutral_state,
                        move_probs,
                        value_target
                    ))
        return return_memory, hMove_counter

    def get_my_engine_move(self, board):
        policy, _ = self.model(
            self.game.get_encoded_state(board).clone().detach().to(self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).detach().numpy()

        validMoves = self.game.get_binary_moves(board)
        policy *= validMoves


        action_probs = policy
        temperature_action_probs = action_probs ** (1 / self.args['temperature'])
        temperature_action_probs /= np.sum(temperature_action_probs)
        action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)
        move = self.game.all_moves[action]
        return move

    def scale_tanh(self, cp):
        k = self.args["tanh_k"]
        return math.tanh(cp / k)

    def scale_sigmoid(self, cp):
        k = self.args["sigmoid_k"]
        return 1 / (1 + math.exp(-cp / k))
    def get_stockfish_eval(self, state):
        self.stockfish.set_fen_position(state.fen())
        stockfish.set_depth(5)
        evaluation = stockfish.get_evaluation()
        if evaluation["type"] == "mate":
            return 1200
        else:
            return evaluation["value"]


    def get_stockfish_move_probs(self, state):
        valid_moves = self.game.get_binary_moves(state)
        move_probs = np.zeros(self.game.actionSize)
        for i, move in enumerate(valid_moves):
            if move !=0:
                new_state = state.copy()
                new_state.push(self.game.all_moves[i])

                move_eval = self.get_stockfish_eval(new_state)
                move_prob = self.scale_sigmoid(move_eval)
                move_probs[move] = move_prob
            else:
                move_probs[move] = 0

        move_probs /= np.sum(move_probs)
        return move_probs

    def train_engine(self, memory):

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            if not sample:
                print(f"Empty sample at batch index {batchIdx}, skipping...")
                continue
            print(f"Sample at batch index {batchIdx}: {sample}")
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)


            out_policy, out_value = self.model(state)

            # Policy loss remains the same
            policy_loss = F.cross_entropy(out_policy, policy_targets)

            # Value loss: train on the average of q and z
            value_loss = F.mse_loss(out_value, value_targets)

            # Total loss
            loss = policy_loss + value_loss

            with open('output.loss_data_stockfish_training', 'a') as f:
                f.write('\n' + str(loss))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def engine_learn(self):
        for iteration in range(self.args['num_iterations']):
            self.model.eval()
            memory = []

            num_batches = self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']

            for selfPlay_iteration in range(num_batches):
                success, returned_memory, num_moves = play_wrapper(selfPlay_iteration)
                if success:
                    memory.extend(returned_memory)


            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train_engine(memory)

            # Save model and optimizer state
            torch.save(self.model.state_dict(), f"model_{iteration}_stockfish_training.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_stockfish_training.pt")

def play_wrapper(i):
    try:
        memory, num_moves = stockfish_train.play_game()
        return True, memory, num_moves
    except:
        print(f"crash at iteration: {i}")
        return False, 0, 0

args = {
    'C': 2,
    'num_searches': 400,
    'num_iterations' : 10,
    'num_selfPlay_iterations' : 100,
    'num_parallel_games' : 1,
    'num_epochs' : 8,
    'batch_size' : 32,
    'temperature' : 1.25,
    'dirichlet_epsilon' : 0.75,
    'dirichlet_alpha' : 0.6,
    'num_engine_games' : 100,
    'num_max_parallel_batches' : 12,
    'num_max_searches' : 500,
    'tanh_k' : 600,
    'sigmoid_k' : 400
}

device = torch.device("cpu")
game = Chess.game
stockfish_path = r"C:\Users\toren\Downloads\stockfish-windows-x86-64-sse41-popcnt\stockfish\stockfish-windows-x86-64-sse41-popcnt"
model = Chess.ResNet(game, 8, 64, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
board = chess.Board()
stockfish = Stockfish(stockfish_path)

stockfish_train = StockfishTrain(game, model, stockfish, optimizer, args)
stockfish_train.engine_learn()