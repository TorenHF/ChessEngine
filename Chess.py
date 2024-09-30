import chess
import numpy as np
import torch
import math
import torch.nn as nn
import torch.functional as F








class Game:
    def __init__(self):
        self.columnCount = 8
        self.rowCount = 8
        self.action_size = 1856
        self.swapDictionary = {
            'R': 'r', 'N': 'n', 'B': 'b', 'Q': 'q', 'K': 'k', 'P': 'p',
            'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K', 'p': 'P',
            '.': '.'
        }

    def changePerspective(self, state, player):
        if player == 1:
            for column in range(self.columnCount):
                for row in range(self.rowCount):
                    state[column][row] = self.swapDictionary[state[column][row]]

        return state

    def getOpponent(self, player):
        player *= -1
        return player

    def getOpponentValue(self, value):
        return -value

    def get_value_and_terminate(self, state, action):
        state.push(action)
        if state.is_checkmate():
            return 1, True
        elif state.is_stalemate():
            return 0, True
        else:
            return 0, False

    def is_move_never_possible(self, move):
        # Extract source and target squares
        source_rank = chess.square_rank(move.from_square)
        source_file = chess.square_file(move.from_square)
        target_rank = chess.square_rank(move.to_square)
        target_file = chess.square_file(move.to_square)

        # Compute rank and file differences
        rank_diff = abs(target_rank - source_rank)
        file_diff = abs(target_file - source_file)

        # A move is impossible if no chess piece can perform it
        # Castling: The king moves two squares horizontally from its starting position
        if move.from_square == chess.E1 and move.to_square in [chess.G1, chess.C1]:  # White castling
            return False  # Valid castling move for White
        if move.from_square == chess.E8 and move.to_square in [chess.G8, chess.C8]:  # Black castling
            return False  # Valid castling move for Black
        # Knight: moves in L shape (2 squares one direction, 1 the other)
        if (rank_diff == 2 and file_diff == 1) or (rank_diff == 1 and file_diff == 2):
            return False  # valid knight move

        # Rook: moves along ranks or files (either rank or file must be the same)
        if (rank_diff == 0 or file_diff == 0):
            return False  # valid rook move

        # Bishop: moves diagonally (rank and file must change by the same amount)
        if rank_diff == file_diff and rank_diff > 0:
            return False  # valid bishop move

        # Queen: combines rook and bishop movement
        if (rank_diff == 0 or file_diff == 0) or (rank_diff == file_diff):
            return False  # valid queen move

        # King: moves 1 square in any direction
        if rank_diff <= 1 and file_diff <= 1:
            return False  # valid king move

        # Pawn: special case, can only move forward or capture diagonally forward
        if source_file == target_file and (rank_diff == 1 or (rank_diff == 2 and source_rank in [1, 6])):
            return False  # valid pawn move forward
        if file_diff == 1 and rank_diff == 1:
            return False  # valid pawn capture diagonally

        # If none of the above conditions are met, the move is never possible
        return True

    def piece_to_vector(self, piece, colour):

        if piece == chess.Piece(chess.PAWN, chess.WHITE):
            return np.array([1,0,0,0,0,0])
        elif piece == chess.Piece(chess.BISHOP, chess.WHITE):
            return np.array([0,1,0,0,0,0])
        elif piece == chess.Piece(chess.KNIGHT, chess.WHITE):
            return np.array([0,0,1,0,0,0])
        elif piece == chess.Piece(chess.ROOK, chess.WHITE):
            return np.array([0,0,0,1,0,0])
        elif piece == chess.Piece(chess.QUEEN, chess.WHITE):
            return np.array([0,0,0,0,1,0])
        elif piece == chess.Piece(chess.KING, chess.WHITE):
            return np.array([0,0,0,0,0,1])
        elif piece is None:
            return np.array([0,0,0,0,0,0])
        elif piece == chess.Piece(chess.PAWN, chess.BLACK):
            return np.array([-1,0,0,0,0,0])
        elif piece == chess.Piece(chess.BISHOP, chess.BLACK):
            return np.array([0,-1,0,0,0,0])
        elif piece == chess.Piece(chess.KNIGHT, chess.BLACK):
            return np.array([0,0,-1,0,0,0])
        elif piece == chess.Piece(chess.ROOK, chess.BLACK):
            return np.array([0,0,0,-1,0,0])
        elif piece == chess.Piece(chess.QUEEN, chess.BLACK):
            return np.array([0,0,0,0,-1,0])
        elif piece == chess.Piece(chess.KING, chess.BLACK):
            return np.array([0, 0, 0, 0, 0, -1])




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
        possibleMoves = self.getAllMoves()
        for move in possibleMoves:
            if board.is_legal(move):
                binaryMoves.append(1)
            else:
                binaryMoves.append(0)
        return binaryMoves


    def get_encoded_state(self, board):
        encoded_state= np.zeros((8, 8, 6))
        for row in range(self.rowCount):
            for column in range(self.columnCount):
                piece = board.piece_at(chess.square(column, row))
                encoded_state[row, column] = self.piece_to_vector(piece, chess.WHITE)
        encoded_state = encoded_state.astype(np.float32)
        return encoded_state





class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

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

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = child_state.push(policy)
                child_state = self.game.changePerspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
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
    def search(self, state, model):
        root = Node(self.game, self.args, state, visit_count=1)
        policy, _ = model(
            torch.tensor(self.game.get_encoded_state(state), device=model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = ((1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon']
                  * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size))

        validMoves = self.game.get_binary_moves(state)
        policy *= validMoves

        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminate(node.state, node.action_taken)
            value = self.game.getOpponentValue(value)

            if not is_terminal:
                policy, value = model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.getValidMoves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

    def findMove(self, state, validMoves):
        action_probs = self.search(state, self.model)
        bestMove = np.argmax(action_probs)
        chessMove = validMoves[bestMove]
        return chessMove

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
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
            nn.Linear(32 * game.rowCount * game.columnCount, game.actionSize)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.rowCount * game.columnCount, 1),
            nn.Tanh()
        )
        self.to(device)

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy,

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
game = Game()
board = chess.Board()
print(board)
list = game.get_binary_moves(board)
print(len(list))
