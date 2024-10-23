import chess
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import ChessTrain
import cProfile
import pstats







class Game:
    def __init__(self, device):
        self.columnCount = 8
        self.rowCount = 8
        self.actionSize = 1856
        self.device = device
        self.swapDictionary = {
            'R': 'r', 'N': 'n', 'B': 'b', 'Q': 'q', 'K': 'k', 'P': 'p',
            'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K', 'p': 'P',
            '.': '.', '1':'1', '2':'2', '3':'3', '4':'4', '5':'5', '6':'6', '7':'7', '8':'8'
        }
        self.all_moves = self.getAllMoves()

    def changePerspective(self, state, player):

        if player == -1:
            # Get the full FEN notation (including piece placement, turn, etc.)
            fen = state.fen()
            # Split FEN into its components (piece placement, turn, etc.)
            parts = fen.split(' ')

            # Split the board portion of FEN into rows and reverse them
            board_rows = parts[0].split('/')
            flipped_rows = board_rows[::-1]

            # Swap piece colors: lowercase (black) ↔ uppercase (white)
            swapped_rows = []

            for row in flipped_rows:
                swapped_row = ""
                for char in row:
                    # Check if the character is a piece and swap its color
                    if char in 'prnbqk':  # Black to white
                        swapped_row += char.upper()
                    elif char in 'PRNBQK':  # White to black
                        swapped_row += char.lower()
                    else:
                        # If it's a number (empty squares), keep it as it is
                        swapped_row += char
                swapped_rows.append(swapped_row)
            # Reassemble the board part of the FEN
            flipped_fen = '/'.join(swapped_rows)

            # Replace the board part with the flipped and swapped version
            parts[0] = flipped_fen

            # Flip the active color ('w' ↔ 'b')
            parts[1] = 'b' if parts[1] == 'w' else 'w'

            # Reconstruct the full FEN and load it into a new board
            flipped_fen = ' '.join(parts)
            state = chess.Board(flipped_fen)


        return state

    def changePerspective_parallel(self, states, player):
        if player == -1:
            for i, state in enumerate(states):
                flipped_state = self.changePerspective(state, player)
                states[i] = flipped_state
        return states

    def flip_move(self, move, player):
        if player == -1:
            flipped_from = chess.square_mirror(move.from_square)
            flipped_to = chess.square_mirror(move.to_square)

            # Create a new move with the flipped squares
            flipped_move = chess.Move(flipped_from, flipped_to)

            # Preserve the promotion piece, if any (e.g., pawn promotion to queen)
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
        elif state.is_stalemate():
            return 0, True
        elif num_moves > 10:
            return 0, True
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

    def piece_to_vector(self, piece):
        if piece == chess.Piece(chess.PAWN, chess.WHITE):
            return torch.tensor([1,0,0,0,0,0])
        elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
            return torch.tensor([0,1,0,0,0,0])
        elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
            return torch.tensor([0,0,1,0,0,0])
        elif piece == chess.Piece(chess.ROOK, chess.WHITE):
            return torch.tensor([0,0,0,1,0,0])
        elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
            return torch.tensor([0,0,0,0,1,0])
        elif piece == chess.Piece(chess.KING, chess.WHITE):
            return torch.tensor([0,0,0,0,0,1])
        elif piece is None:
            return torch.tensor([0,0,0,0,0,0])
        elif piece == chess.Piece(chess.PAWN, chess.BLACK):
            return torch.tensor([-1,0,0,0,0,0])
        elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
            return torch.tensor([0,-1,0,0,0,0])
        elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
            return torch.tensor([0,0,-1,0,0,0])
        elif piece == chess.Piece(chess.ROOK, chess.BLACK):
            return torch.tensor([0,0,0,-1,0,0])
        elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
            return torch.tensor([0,0,0,0,-1,0])
        elif piece == chess.Piece(chess.KING, chess.BLACK):
            return torch.tensor([0, 0, 0, 0, 0, -1])




    def getAllMoves(self):
        moves = []
        for source_square in chess.SQUARES:
            for target_square in chess.SQUARES:
                move = chess.Move(source_square, target_square)
                if not self.is_move_never_possible(move):
                    moves.append(move)
        return moves

    def get_binary_moves(self, board):
        binaryMoves = []
        board.turn = chess.WHITE
        for move in self.all_moves:
            if board.is_legal(move):
                binaryMoves.append(1)
            else:
                binaryMoves.append(0)
        return binaryMoves



    def get_encoded_state(self, board):
        encoded_state = torch.zeros((8, 8, 6), dtype=torch.float32)

        for row in range(self.rowCount):
            for column in range(self.columnCount):
                piece = board.piece_at(chess.square(column, row))
                sourceTensor = self.piece_to_vector(piece)
                encoded_state[row, column] = sourceTensor.clone().detach()


        return encoded_state.to(self.device)

    def get_encoded_state_parallel(self, states):
        encoded_states = torch.zeros((int(len(states)), 8, 8, 6), dtype=torch.float32)
        for idx, state in enumerate(states):
            #encoded_state = torch.zeros((8, 8, 6), dtype=torch.float32)
            for row in range(self.rowCount):
                for column in range(self.columnCount):
                    piece = state.piece_at(chess.square(column, row))
                    sourceTensor = self.piece_to_vector(piece)
                    #encoded_state[row, column] = sourceTensor.clone().detach()
                    encoded_states[idx, row, column] = sourceTensor.clone().detach()

        return encoded_states.to(self.device)

    def play(self, state, player):
        while True:
            print(state)
            if player == 1:
                valid_moves = state.legal_moves
                #print("Valid moves:", [i for i in range(self.action_size) if valid_moves[i] == 1])
                print(state.legal_moves)
                action = str(input(f"Player {player}, enter your move: "))

                #if valid_moves[action] == 0:
                    #print("Action not valid. Try again.")
                    #continue
                state.push_san(action)
            else:
                neutral_state = self.changePerspective(state, player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)
                move = self.all_moves[action]
                neutral_state.push(move)
                state = self.changePerspective(neutral_state, player)


            value, is_terminal = self.get_value_and_terminate(state)

            if is_terminal:
                print(state)
                if value == 1:
                    print(f"Player {player} won!")
                else:
                    print("It's a draw!")
                break

            player = self.getOpponent(player)





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

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        return best_child

    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (1 + child.visit_count)) * child.prior

    def expand(self, policy, player):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                move = self.game.all_moves[action]

                child_state.push(move)

                child_state = self.game.changePerspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, move, action, prob)
                self.children.append(child)

        return child

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

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.getOpponentValue(value)
        if self.parent is not None:
            self.parent.backpropagate(value)

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
        policy, _ = model(
            self.game.get_encoded_state(state).clone().detach().to(model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.actionSize))

        validMoves = self.game.get_binary_moves(state)
        policy *= validMoves

        policy /= np.sum(policy)
        root.expand(policy, player)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminate(node.state)
            value = self.game.getOpponentValue(value)

            if not is_terminal:
                policy, value = model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=model.device).unsqueeze(0)
                )
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
        return action_probs

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(8, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 6, game.actionSize)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 8 * 6, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value

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

args = {
    'C': 2,
    'num_searches': 500,
    'num_iterations' : 1,
    'num_selfPlay_iterations' : 1,
    'num_parallel_games' : 1,
    'num_epochs' : 4,
    'batch_size' : 64,
    'temperature' : 1.25,
    'dirichlet_epsilon' : 0.25,
    'dirichlet_alpha' : 0.3,
    'num_engine_games' : 100
}
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
game = Game(device)
player = 1

state = chess.Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR')
move = chess.Move.from_uci('e2e3')



model = ResNet(game, 5, 128, device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
mcts = MCTS(args, state, player, game, model)

alphazero = ChessTrain.AlphaZeroParallel(model, optimizer, game, args, Node)
profiler = cProfile.Profile()

profiler.enable()

# Run the function you want to profile
alphazero.learn()

profiler.disable()
profiler.dump_stats('output.prof')  # Save to a file

# Load and view stats
stats = pstats.Stats('output.prof')
stats.strip_dirs().sort_stats('time').print_stats(10)  # Show top 10 functions by time
