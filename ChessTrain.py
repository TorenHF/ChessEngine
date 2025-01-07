import numpy as np
import torch
import torch.nn.functional as F
import random
import chess
import torch.multiprocessing as mp
from functools import partial
import time
import os

"""
    This code is for the engine in the paper: Creating an Engine that teaches Itself to Play Chess, by Toren Hewitt-Fry, 2025.
    The code in this file was partially created using AI sources.
    All functions in this code, which were made using assistance from AI, cite the prompt number, which corresponds to a prompt in my Paper in Section 11.1.
    All functions in this code, which correspond to certain Sections within my Paper, reference those Sections.
    """

# This class is for the self-play training of my model, which is referenced in Section: 8.6
class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args, node, mcts):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = mcts
        self.state = chess.Board()
        self.root = None
    # This function is for the training of my model based on the results gathered in the self-play games
    def train(self, memory):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:min(len(memory) -1, batchIdx+self.args['batch_size'])]

            state, policy_targets, value_targets, q_values = zip(*sample)


            q_values = np.array(q_values).reshape(-1, 1)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)

            avg_qz = (q_values + value_targets) / 2.0

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            avg_qz = torch.tensor(avg_qz, dtype=torch.float32, device=self.model.device)
            value_targets_tensor = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)

            value_loss = F.mse_loss(out_value, avg_qz)

            loss = policy_loss + 2*value_loss
            loss_np = loss.cpu().detach().numpy()
            value_loss_np = value_loss.cpu().detach().numpy()

            with open('output.value-loss_data.complete_restart', 'a') as f:
                f.write('\n' + str(value_loss_np))

            with open('output.loss_data.complete_restart', 'a') as f:
                f.write('\n' + str(loss_np))



            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    # This function is referenced in Section 8.6.5.3
    def train_on_wins(self, memory):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model.to(self.device)

        random.shuffle(memory)

        for batchIdx in range(0, len(memory), self.args['wins_batch_size']):
            sample = memory[batchIdx:min(len(memory) - 1, batchIdx + self.args['wins_batch_size'])]

            state, policy_targets, value_targets, q_values = zip(*sample)

            q_values = np.array(q_values).reshape(-1, 1)

            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(
                value_targets).reshape(-1, 1)

            avg_qz = (q_values + value_targets) / 2.0

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            avg_qz = torch.tensor(avg_qz, dtype=torch.float32, device=self.model.device)
            value_targets_tensor = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)

            value_loss = F.mse_loss(out_value, avg_qz)

            loss = policy_loss + 2*value_loss
            loss_np = loss.cpu().detach().numpy()
            value_loss_np = value_loss.cpu().detach().numpy()

            with open('output.value-loss_data.onlyWins.complete_restart', 'a') as f:
                f.write('\n' + str(value_loss_np))

            with open('output.loss_data.onlyWins.complete_restart', 'a') as f:
                f.write('\n' + str(loss_np))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    # This section runs both the gathering of the self-play games and the training of the model based on those results
    # This function was written / optimized with the help of AI and corresponds to AI prompt: 4, 5, 6, 7
    def learn(self, test):
        # Move the model to CPU to make it picklable
        mp.set_start_method('spawn', force=True)
        self.device = torch.device("cpu")
        self.model.to(self.device)


        max_games = self.args['num_max_parallel_batches'] * self.args['num_parallel_games'] + self.args['num_parallel_games']
        start_parameter_mcts = self.args['num_searches'] - self.args['num_max_searches']
        start_parameter_spg = self.args['num_parallel_games'] - max_games

        with mp.Pool(processes=self.args['num_parallel_games']) as pool:
            for iteration in range(self.args['num_iterations']):
                self.model.eval()
                memory = []
                win_memory = []

                #self.args['num_searches'] =  round(float(start_parameter_mcts * np.exp(-(iteration + 14) * 0.5) + self.args['num_max_searches']))
                #self.args['num_selfPlay_iterations'] = round(float(start_parameter_spg * np.exp(-(iteration + 19) * 0.5) + max_games))
                # removed these for the end of training as the iteration was so high that both functions had reached their limits


                selfPlay_partial = partial(selfPlay_wrapper, self.mcts, self.game, self.args, self.model,
                                           self.model.state_dict(), test)


                num_batches = self.args['num_selfPlay_iterations'] // self.args['num_parallel_games']



                for selfPlay_iteration in range(num_batches):
                    results = pool.map(selfPlay_partial, range(self.args['num_parallel_games']))

                    for idx, (success, result, num_moves, winner) in enumerate(results):
                        if success:
                            game_memory = result
                            memory.extend(game_memory)
                            if winner !=0:
                                win_memory.extend(game_memory)
                            with open('output.num_moves.data.complete_restart', 'a') as f:
                                f.write('\n' + str(num_moves))
                            with open('output.results.complete_restart', 'a') as f:
                                f.write('\n' + str(winner))

                    if test:
                        with open('start_logs.data', 'w') as outfile:
                            for i in range(self.args['num_parallel_games']):
                                log_filename = f"worker_{i}_start_log.data"
                                with open(log_filename, 'r') as infile:
                                    outfile.write(infile.read())
                        cleanup_logs(self.args['num_parallel_games'])

                self.model.train()
                for epoch in range(self.args['num_epochs']):
                    self.train(memory)
                for win_epoch in range(self.args['num_win_epochs']):
                    self.train_on_wins(win_memory)

                self.model.to('cpu')

                torch.save(self.model.state_dict(), f"model_{iteration}_complete_restart.pt")
                torch.save(self.optimizer.state_dict(), f"optimizer_{iteration}_complete_restart.pt")


def cleanup_logs(num_workers):
    for i in range(num_workers):
        log_filename_1 = f"worker_{i}_start_log.data"
        log_filename_2 = f"worker_{i}_move_log.data"
        if os.path.exists(log_filename_1):
            os.remove(log_filename_1)
        if os.path.exists(log_filename_2):
            os.remove(log_filename_2)


def selfPlay_wrapper(mcts, game, args, model, model_state_dict, test, i):
    try:

        # Initialize device
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if test:
            with open(f"worker_{i}_start_log.data", 'a') as f:
                f.write(f'\n started at {time.localtime()}')

        model.load_state_dict(model_state_dict)

        model.to(device)
        model.eval()

        result, num_moves, winner = selfPlay(game, mcts, args, i, test)
        if test:
            with open(f"worker_{i}_move_log.data", 'a') as f:
                f.write(f'\n finished at {time.localtime()}')

        return (True, result, num_moves, winner)

    except Exception as e:
        print(f"Worker {i}: Exception occurred: {e}")
        return (False, e, 0, 0)
# This is the function for the playing of self-play games and is referenced throughout my whole paper, especially in Chapter 8.6
def selfPlay(game, mcts, args, i, test):
    return_memory = []
    memory = []
    player = 1
    value = 0
    state = chess.Board()
    num_moves = 0
    is_terminal = False

    while not is_terminal:
        num_moves += 1

        if test:
            with open(f"worker_{i}_move_log.data", 'a') as f:
                f.write(f'\n {num_moves} played at {time.localtime()}')

        neutral_state = game.changePerspective(state, player)
        action_probs, root = mcts.search(neutral_state)

        temperature_action_probs = action_probs ** (1 / args['temperature'])
        temperature_action_probs /= np.sum(temperature_action_probs)

        action = np.random.choice(len(temperature_action_probs), p=temperature_action_probs)
        move = game.all_moves[action]
        move = game.flip_move(move, player)
        state.push(move)

        value, is_terminal = game.get_value_and_terminate(state, num_moves)


        q_value = root.value_sum / root.visit_count
        memory.append((neutral_state, action_probs, player, q_value))

        if is_terminal:
            for hist_neutral_state, hist_action_probs, hist_player, hist_q_value in memory:
                hist_outcome = value if hist_player == player else game.getOpponentValue(value)
                return_memory.append((
                    game.get_encoded_state(hist_neutral_state).cpu(),
                    hist_action_probs,
                    hist_outcome,
                    hist_q_value
                ))

        player = game.getOpponent(player)
    winner = value if player == 1 else game.getOpponent(value)
    return return_memory, num_moves, winner


# This function corresponds to the function in Section 8.6.5.2
# This function was written / optimized with the help of AI and corresponds to AI prompt: 1, 2, 3
def generate_random_chess_position(max_attempts=1000):
    """
    Generates a random, non-terminal chess position using the python-chess library.

    Args:
        max_attempts (int): Maximum number of attempts to generate a non-terminal position.

    Returns:
        chess.Board: A randomly generated, non-terminal chess board position.

    Raises:
        ValueError: If a non-terminal position could not be generated within max_attempts.
    """
    piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
    max_pieces = {
        chess.PAWN: 8,
        chess.KNIGHT: 10,  # Including promotions
        chess.BISHOP: 10,
        chess.ROOK: 10,
        chess.QUEEN: 9,
    }

    for attempt in range(max_attempts):
        board = chess.Board(None)  # Start with an empty board

        # Helper function to get empty squares
        def get_empty_squares(for_pawn=False):
            empty = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
            if for_pawn:
                # Exclude first and eighth ranks for pawns
                empty = [sq for sq in empty if chess.square_rank(sq) not in [0, 7]]
            return empty

        # Helper function to randomly place a single king
        def place_king(color):
            empty = get_empty_squares()
            king_square = random.choice(empty)
            board.set_piece_at(king_square, chess.Piece(chess.KING, color))

        # Place white and black kings
        place_king(chess.WHITE)
        place_king(chess.BLACK)

        # Function to randomly place pieces for a given color
        def place_pieces(color):
            remaining_pieces = {}
            for pt in piece_types:
                remaining_pieces[pt] = random.randint(0, max_pieces[pt])

            # Place pieces
            for piece_type, count in remaining_pieces.items():
                for _ in range(count):
                    if piece_type == chess.PAWN:
                        empty = get_empty_squares(for_pawn=True)
                    else:
                        empty = get_empty_squares()
                    if not empty:
                        break  # No more space to place pieces
                    square = random.choice(empty)
                    board.set_piece_at(square, chess.Piece(piece_type, color))

        # Place pieces for both colors
        place_pieces(chess.WHITE)
        place_pieces(chess.BLACK)

        # Randomly set side to move
        board.turn = random.choice([chess.WHITE, chess.BLACK])

        # Randomly set castling rights
        castling_rights = 0
        # Castling rights are represented by constants in python-chess
        # Check if kings and rooks are on their starting squares before granting rights

        # White kingside
        if (board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and
            board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE)):
            if random.choice([True, False]):
                castling_rights |= chess.BB_H1
        # White queenside
        if (board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE) and
            board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE)):
            if random.choice([True, False]):
                castling_rights |= chess.BB_A1
        # Black kingside
        if (board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and
            board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK)):
            if random.choice([True, False]):
                castling_rights |= chess.BB_H8
        # Black queenside
        if (board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK) and
            board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK)):
            if random.choice([True, False]):
                castling_rights |= chess.BB_A8

        board.castling_rights = castling_rights

        # Randomly set en passant square
        if random.choice([True, False]):
            # Choose a file from a to h and a rank from 3 to 6 (0-indexed: 2 to 5)
            ep_file = random.choice(range(8))
            ep_rank = random.choice([2, 3, 4, 5])
            ep_square = chess.square(ep_file, ep_rank)
            # Optionally, ensure that a pawn is in position to perform en passant
            # For simplicity, we'll skip this validation
            board.ep_square = ep_square
        else:
            board.ep_square = None

        # Set halfmove clock and fullmove number randomly
        board.halfmove_clock = random.randint(0, 100)
        board.fullmove_number = random.randint(1, 200)

        # Check if the position is terminal
        if not board.is_game_over():
            return board
        # Else, continue to the next attempt

    # If no non-terminal position was found within max_attempts
    raise ValueError(f"Failed to generate a non-terminal position in {max_attempts} attempts.")
