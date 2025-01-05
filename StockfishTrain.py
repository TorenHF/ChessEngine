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
            eval = self.get_stockfish_eval(flipped_board)
            move_probs = self.get_stockfish_move_probs(flipped_board)


            memory.append((neutral_state, copy_board, eval, move_probs))

            if hMove_counter % 4 == 0 or hMove_counter % 3 == 0:
                legal_moves = self.game.get_binary_moves(flipped_board)
                move_probs = legal_moves / np.sum(legal_moves)
                action = np.random.choice(len(self.game.all_moves), p=move_probs)
                move = self.game.all_moves[action]

            else:
                for i, action in enumerate(move_probs):
                    if action == 1:
                        move = self.game.all_moves[i]
            move = self.game.flip_move(move, player)

            board.push(move)

            hMove_counter += 1
            value, is_terminal = self.game.get_value_and_terminate(board, hMove_counter)
            player = self.game.getOpponent(player)


            if is_terminal:
                for hist_neutral_state, hist_board, hist_eval, hist_move_probs in memory:
                    value_target = self.scale_tanh(hist_eval)
                    return_memory.append((
                        hist_neutral_state.cpu().detach().numpy(),
                        hist_move_probs,
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
        stockfish.set_depth(15)
        evaluation = stockfish.get_evaluation()
        if evaluation["type"] == "mate":
            return 1200 if evaluation["value"] > 0 else -1200
        else:
            return evaluation["value"]


    def get_stockfish_move_probs(self, state):
        move_probs = np.zeros(self.game.actionSize)
        self.stockfish.set_fen_position(state.fen())
        move = self.stockfish.get_best_move_time(600)
        move = chess.Move.from_uci(move)
        action = self.game.all_moves.index(move)
        move_probs[action] = 1

        return move_probs

    def train_engine(self, memory):

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['batch_size'])]
            if not sample:
                print(f"Empty sample at batch index {batchIdx}, skipping...")
                continue

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

            loss_np = loss.cpu().detach().numpy()
            value_loss_np = value_loss.cpu().detach().numpy()

            with open('output.value-loss_data.stockfish_training', 'a') as f:
                f.write('\n' + str(value_loss_np))

            with open('output.loss_data.stockfish_training', 'a') as f:
                f.write('\n' + str(loss_np))
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
                    with open('output.num_moves.stockfish', 'a') as f:
                        f.write('\n' + str(num_moves))


            self.model.train()
            for epoch in range(self.args['num_epochs']):
                self.train_engine(memory)

            # Save model and optimizer state
            torch.save(self.model.state_dict(), f"model_{iteration+3}_stockfish_training.pt")
            torch.save(self.optimizer.state_dict(), f"optimizer_{iteration+3}_stockfish_training.pt")

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
    'num_iterations' : 40,
    'num_selfPlay_iterations' : 800,
    'num_parallel_games' : 1,
    'num_epochs' : 4,
    'batch_size' : 128,
    'temperature' : 2,
    'tanh_k' : 600,
    'sigmoid_k' : 400
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
game = Chess.game
stockfish_path = "/opt/homebrew/bin/stockfish"
model = Chess.ResNet(game, 12, 128, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
board = chess.Board()
stockfish = Stockfish(stockfish_path, parameters={
    "Threads" : 10,
    "Hash" : 512
})
state_dict = torch.load('model_4_stockfish_training.pt')
new_optimizer_state_dict = torch.load('optimizer_4_stockfish_training.pt')
model.load_state_dict(state_dict)
optimizer.load_state_dict(new_optimizer_state_dict)
stockfish_train = StockfishTrain(game, model, stockfish, optimizer, args)

stockfish_train.engine_learn()