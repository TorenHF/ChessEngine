import chess
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import ChessTrain
import cProfile

"""
    This code is for the engine in the paper: Creating an Engine that teaches Itself to Play Chess, by Toren Hewitt-Fry, 2025.
    The code in this file was partially created using AI sources.
    All functions in this code, which were made using assistance from AI, cite the prompt number, which corresponds to a prompt in my Paper in Section 11.1.
    All functions in this code, which correspond to certain Sections within my Paper, reference those Sections.
    """


# This class is for the functions that make my self-play and MCTS methods work.
class Game:
    def __init__(self, device):
        self.columnCount = 8
        self.rowCount = 8
        self.device = device
        self.swapDictionary = {'R': 'r', 'N': 'n', 'B': 'b', 'Q': 'q', 'K': 'k', 'P': 'p', 'r': 'R', 'n': 'N', 'b': 'B',
            'q': 'Q', 'k': 'K', 'p': 'P', '.': '.', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
            '7': '7', '8': '8'}
        self.swap_pieces = {'p': 'P', 'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K', 'P': 'p', 'R': 'r', 'N': 'n',
            'B': 'b', 'Q': 'q', 'K': 'k'}
        self.swapDictionary_pieces = {}
        self.all_moves = self.getAllMoves()
        self.actionSize = len(self.all_moves)

    # This function corresponds to the function in Section 8.2.1
    # This function was written / optimized with the help of AI and corresponds to AI prompt: 11, 12
    def changePerspective(self, state, player):
        if player == -1:
            fen = state.fen()
            parts = fen.split(' ')
            board_rows = parts[0].split('/')

            flipped_fen = '/'.join(
                ''.join(self.swap_pieces.get(char, char) for char in row) for row in board_rows[::-1])

            parts[0] = flipped_fen

            parts[1] = 'b' if parts[1] == 'w' else 'w'

            flipped_fen = ' '.join(parts)
            state = chess.Board(flipped_fen)

        return state

    # This function is no longer used, because of the implementation of multiprocessing, but is referenced in Section 8.4 and Figure 29
    def changePerspective_parallel(self, states, player):
        if player == -1:
            for i, state in enumerate(states):
                flipped_state = self.changePerspective(state, player)
                states[i] = flipped_state
        return states

    # This function corresponds to the function in Section 8.2.2
    def flip_move(self, move, player):
        if player == -1:
            flipped_from = chess.square_mirror(move.from_square)
            flipped_to = chess.square_mirror(move.to_square)
            flipped_move = chess.Move(flipped_from, flipped_to)

            if move.promotion:
                flipped_move.promotion = move.promotion
        else:
            flipped_move = move
        return flipped_move

    def getOpponent(self, player):
        player *= -1
        return player

    def getOpponentValue(self, value):
        return -value

    def get_value_and_terminate(self, state, num_moves):
        if state.is_checkmate():
            return 1, True
        elif state.is_stalemate() or state.is_insufficient_material() or state.is_repetition(3):
            return 0, True
        elif num_moves > 567:
            return 0, True
        else:
            return 0, False

    def get_value_and_terminate_mcts(self, state):
        if state.is_checkmate():
            return 1, True
        elif state.is_stalemate() or state.is_insufficient_material() or state.is_repetition(3):
            return -0.05, True #slightly discourage the MCTS from playing draws
        else:
            return 0, False

    def is_move_never_possible(self, move):
        source_rank = chess.square_rank(move.from_square)
        source_file = chess.square_file(move.from_square)
        target_rank = chess.square_rank(move.to_square)
        target_file = chess.square_file(move.to_square)

        rank_diff = abs(target_rank - source_rank)
        file_diff = abs(target_file - source_file)

        if move.from_square == chess.E1 and move.to_square in [chess.G1, chess.C1]:  # White castling
            return False  # Valid castling move for White
        if move.from_square == chess.E8 and move.to_square in [chess.G8, chess.C8]:  # Black castling
            return False  # Valid castling move for Black
        if (rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2):
            return False  # valid knight move
        if (rank_diff == 0 or file_diff == 0):
            return False  # valid rook move
        if rank_diff == file_diff and rank_diff > 0:
            return False  # valid bishop move
        if (rank_diff == 0 or file_diff == 0) or (rank_diff == file_diff):
            return False  # valid queen move
        if rank_diff <= 1 and file_diff <= 1:
            return False  # valid king move
        if source_file == target_file and (rank_diff == 1 or (rank_diff == 2 and source_rank in [1, 6])):
            return False  # valid pawn move forward
        if file_diff == 1 and rank_diff == 1:
            return False  # valid pawn capture diagonally

        return True


    # This function was written / optimized with the help of AI and corresponds to AI prompt: 8, 13
    def piece_to_vector(self, piece):
        if piece is None:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.PAWN, chess.BLACK):
            return np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.PAWN, chess.WHITE):
            return np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.ROOK, chess.BLACK):
            return np.array([0, 0, 0, -1, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.ROOK, chess.WHITE):
            return np.array([0, 0, 0, 1, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
            return np.array([0, 1, 0, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
            return np.array([0, 0, 1, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
            return np.array([0, -1, 0, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
            return np.array([0, 0, -1, 0, 0, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.KING, chess.BLACK):
            return np.array([0, 0, 0, 0, 0, -1], dtype=np.float32)
        elif piece == chess.Piece(chess.KING, chess.WHITE):
            return np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
        elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
            return np.array([0, 0, 0, 0, 1, 0], dtype=np.float32)
        elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
            return np.array([0, 0, 0, 0, -1, 0], dtype=np.float32)

    # This function corresponds to the function in Section 8.2.3
    # This function was written / optimized with the help of AI and corresponds to AI prompt: 8, 13
    def get_encoded_state(self, board):
        encoded_state_np = np.zeros((8, 8, 6), dtype=np.float32)

        for row in range(self.rowCount):
            for column in range(self.columnCount):
                piece = board.piece_at(chess.square(column, row))
                encoded_state_np[row, column] = self.piece_to_vector(piece)

        encoded_state = torch.tensor(encoded_state_np, dtype=torch.float32).to(self.device)
        return encoded_state

    # This function is no longer used, because of the implementation of multiprocessing, but is referenced in Section 8.4 and Figure 29
    def get_encoded_state_parallel(self, states):
        encoded_states_np = np.zeros((int(len(states)), 8, 8, 6), dtype=np.float32)
        for idx, state in enumerate(states):
            for row in range(self.rowCount):
                for column in range(self.columnCount):
                    piece = state.piece_at(chess.square(column, row))
                    encoded_state_np = self.piece_to_vector(piece)
                    encoded_states_np[idx, row, column] = encoded_state_np

        encoded_states = torch.tensor(encoded_states_np, dtype=torch.float32).to(self.device)
        return encoded_states

    def getAllMoves(self):
        moves = []
        for source_square in chess.SQUARES:
            for target_square in chess.SQUARES:
                move = chess.Move(source_square, target_square)
                if not self.is_move_never_possible(move):
                    moves.append(move)
        columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        for column in range(8):
            moves.append(chess.Move.from_uci(f'{columns[column]}7{columns[column]}8q'))
            moves.append(chess.Move.from_uci(f'{columns[column]}7{columns[column]}8n'))
        return moves

    # This function corresponds to the function in Section 8.2.4.1
    # This function was written / optimized with the help of AI and corresponds to AI prompt: 9, 10,
    def get_binary_moves(self, board):
        board.turn = chess.WHITE
        binaryMoves = [1 if board.is_legal(move) else 0 for move in self.all_moves]
        return binaryMoves

    def tryMove(self, format):
        if format == 0:
            print("Legal moves: ", self.state.legal_moves)
        else:
            print("Legal moves: ", end="")
            for move in self.state.legal_moves:
                print(move.uci(), end=", ")
            print()

        move = str(input(f"Player {player}, enter your move: "))

        try:
            if format == 0:
                self.state.push_san(move)
            else:
                self.state.push_uci(move)
        except:
            print("Invalid move")
            self.tryMove(format)



    def play(self, state, player):
        print("This is an explanation for how to input the moves for the chess game")
        print("The board is splitt into 8 rows and 8 columns")
        print("Each column is represented by a letter from a to h")
        print("Each row is represented by a number from 1 to 8")
        print("The bottom left square is a1, the top right square is h8")
        success = False
        while not success:
            type = int(input("Type 1 for uci, type 0 for san format for the playing of moves: "))
            if type == 1:
                print("To play a move, you have to list the starting square and ending square of the piece playing the move")
                print("For pawn promotions, the promoted to piece needs to be added onto the end of the move with:")
                print("Queen = q, Bishop = b, Rook = r, Knight = n")
                print("Examples of moves are: a2a3, b6c5, h7h8q (pawn promotion to a queen)")
                format = 1
                success = True
            if type == 0:
                print("To play a move, you have to list the starting piece with its corresponding letter, and the end square of the piece playing the move")
                print("The pieces are: King = K, Queen = Q, Bishop = B, Rook = R, Knight = N, and for pawn moves no piece is listed")
                print("For special moves such as Captures or Checks, special symbols are required:")
                print("Captures = x, Checks = +, Checkmate = #, Pawn promotion = =(Piece)")
                print("The x for captures comes between the piece type and the end sqaure, the rest of the additional information comes at the end of the move")
                print("Examples of moves are: Rh3, Nxb5, Qh2+, e8=R+ (pawn promotion to a rook with check)")
                format = 0
                success = True
            else:
                print("Invalid option")

        print("Good Luck!")
        print("Game is Starting:")
        while True:
            self.state = state
            print(self.state.unicode())
            if player == 1:
                self.tryMove(format)

            else:
                neutral_state = self.changePerspective(self.state, player)
                mcts_probs, _ = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
                move = self.all_moves[action]
                move = self.flip_move(move, player)
                state.push(move)
                print(f"Engine Move: {move}")

            value, is_terminal = self.get_value_and_terminate(self.state, 1)

            if is_terminal:
                print(self.state.unicode())
                if value == 1:
                    print(f"Player {player} won!")
                else:
                    print("It's a draw!")
                break

            player = self.getOpponent(player)

# This class is for the Nodes in the MCTS algorithm
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, action_index=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior
        self.action_index = action_index

        self.children = []
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0
    # This function corresponds to the method described in Sections 6.1.1.1 and 6.1.2
    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    # This function corresponds to the method described in Sections 6.1.1.1 and 6.1.2
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (1 + child.visit_count)) * child.prior

    # This function corresponds to the method described in Sections 6.1.1.2 and 6.1.2
    # This function is also briefly mentioned in Section 8.4 and Figure 29
    def expand(self, policy, player):
        swapped_state = self.game.changePerspective(self.state, player=-1)
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = swapped_state.copy()
                move = self.game.all_moves[action]
                flipped_move = self.game.flip_move(move, player=-1)
                child_state.push(flipped_move)

                child = Node(self.game, self.args, child_state, self, move, action, prob)
                self.children.append(child)
        return child
    # This function is not used for this engine, but corresponds to the method described in Section 6.1.1.2
    def simulate(self):
        value, is_terminal = self.game.get_value_and_terminate(self.state, self.action_taken)
        value = self.game.getOpponentValue(value)
        if is_terminal:
            return value

        rollout_state = self.state.copy()
        rollout_player = 1
        while True:
            valid_moves = rollout_state.valid_moves
            action = np.random.choice(np.where(valid_moves == 1)[0])
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminate(rollout_state, action)
            if is_terminal:
                if rollout_player == -1:
                    value = self.game.getOpponentValue(value)
                return value
            rollout_player = self.game.getOpponent(rollout_player)

    # This function corresponds to the method described in Sections 6.1.1.3 and 6.1.2
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.getOpponentValue(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

# This class is for the MCTS algorithm, which is described in Section 6.1
class MCTS:
    def __init__(self, args, state, player, game, model):
        self.args = args
        self.state = state
        self.player = player
        self.game = game
        self.model = model

    @torch.no_grad()
    def search(self, state):
        model = self.model
        root = Node(self.game, self.args, state, visit_count=1)
        policy, _ = model(self.game.get_encoded_state(state).clone().detach().to(model.device).unsqueeze(0))
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] * np.random.dirichlet(
            [self.args['dirichlet_alpha']] * self.game.actionSize))

        validMoves = self.game.get_binary_moves(state)
        policy *= validMoves

        policy /= np.sum(policy)
        root.expand(policy, player)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminate_mcts(node.state)
            value = self.game.getOpponentValue(value)

            if not is_terminal:
                policy, value = model(self.game.get_encoded_state(state).clone().detach().to(model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_binary_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy, player)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.actionSize)
        for child in root.children:
            action_probs[child.action_index] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs, root

# This class is for the neural network, which is described in Section 6.3
class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(nn.Conv2d(8, num_hidden, kernel_size=3, padding=1), nn.BatchNorm2d(num_hidden),
            nn.ReLU())
        self.backBone = nn.ModuleList([ResBlock(num_hidden) for i in range(num_resBlocks)])
        self.policyHead = nn.Sequential(nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.Flatten(), nn.Linear(32 * 8 * 6, game.actionSize))

        self.valueHead = nn.Sequential(nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1), nn.BatchNorm2d(3), nn.ReLU(),
            nn.Flatten(), nn.Linear(3 * 8 * 6, 1), nn.Tanh())
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

# This class is for the ResBlocks within the neural network, and is described in Section 6.3.4
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


args = {'C': 1, 'num_searches': 1000, 'num_iterations': 100, 'num_selfPlay_iterations': 385, 'num_parallel_games': 11,
    'num_epochs': 4, 'num_win_epochs': 2, 'batch_size': 256, 'wins_batch_size': 64, 'temperature': 2,
    'dirichlet_epsilon': 0.5, 'dirichlet_alpha': 0.6, 'num_engine_games': 100, 'num_max_parallel_batches': 20,
    'num_max_searches': 400

}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
game = Game(device)
player = 1


state = chess.Board()
move = chess.Move.from_uci('e2e3')

self_play_model_state_dict = torch.load('model_59_complete_restart.pt', map_location=device)
stockfish_model_state_dict = torch.load('model_5_stockfish_training.pt', map_location=device)


self_play_model = ResNet(game, 8, 64, device)
stockfish_model = ResNet(game, 12, 128, device)

optimizer = torch.optim.Adam(self_play_model.parameters(), lr=0.001, weight_decay=0.0001)

self_play_model.load_state_dict(self_play_model_state_dict)
stockfish_model.load_state_dict(stockfish_model_state_dict)


profiler = cProfile.Profile() # Using the profiler was done with the help of AI, see AI prompt: 14

#To play against the self-play model, change the variable "stockfish_model" to "self_play_model" below in the MCTS class
mcts = MCTS(args, state, player, game, stockfish_model)

alphazero = ChessTrain.AlphaZeroParallel(self_play_model, optimizer, game, args, Node, mcts)

if __name__ == '__main__':
    game.play(state, player)
