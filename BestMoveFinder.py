import random
import ChessEngineAdv as ChessEngine
import pygame as p

CHECKMATE = 10000000
STALEMATE = 0
DEPTH = 2

piece_score = {"R": 563, "N": 305, "B": 333, "Q": 950, "K": 0, "p": 100}

rook_scores = [[0,  0,  0,  5,  5,  0,  0,  0],
               [5, 10, 15, 15, 15, 15, 10,  5],
               [-5, 0,  0,  0,  0,  0,  0, -5],
               [-5, 0,  0,  0,  0,  0,  0, -5],
               [-5, 0,  0,  0,  0,  0,  0, -5],
               [-5, 0,  0,  0,  0,  0,  0, -5],
               [-5, 0,  0,  5,  5,  0,  0, -5],
               [-15,-10,0, 10, 10,  5,-10, -15]]

knight_scores = [[-50, -25, -25, -25, -25, -25, -25, -50],
                 [-40, -20,   0,   0,   0,   0, -20, -25],
                 [-25,   0,  10,  10,  10,  10,   0, -25],
                 [-25,   5,  10,  15,  15,  10,   5, -25],
                 [-25,   0,  10,  15,  15,  10,   0, -25],
                 [-25,   5,  10,  10,  10,  10,   5, -25],
                 [-40, -20,   0,   5,   5,   0, -20, -40],
                 [-50, -25, -25, -25, -25, -25, -25, -50]]

bishop_scores = [[-20, -10, -10, -10, -10, -10, -10, -20],
                 [-10,   0,   0,   0,   0,   0,   0, -10],
                 [-10,   0,   5,  10,  10,   5,   0, -10],
                 [-10,   5,   5,  10,  10,   5,   5, -10],
                 [-10,   0,  10,  10,  10,  10,   0, -10],
                 [-10,  10,  10,  10,  10,  10,  10, -10],
                 [-10,   5,   0,   0,   0,   0,   5, -10],
                 [-20, -10, -15, -10, -10, -15, -10, -20]]

queen_scores = [[-20, -10, -10, -5, -5, -10, -10, -20],
                [-10,   0,   0,  0,  0,   0,   0, -10],
                [-10,   0,   5,  5,  5,   5,   0, -10],
                [-5,    0,   5,  5,  5,   5,   0,  -5],
                [-5,    0,   5,  5,  5,   5,   0,  -5],
                [-10,   0,   5,  5,  5,   5,   0, -10],
                [-10,   0,   0,  0,  0,   0,   0, -10],
                [-20, -10, -10, -5, -5, -10, -10, -20]]

king_scores = [[-10, -10, -10, -20, -20, -10, -10, -10],
               [-30, -30, -40, -40, -40, -40, -30, -30],
               [-30, -40, -40, -50, -50, -40, -40, -30],
               [-30, -40, -40, -50, -50, -40, -40, -30],
               [-20, -30, -30, -40, -40, -30, -30, -20],
               [-10, -20, -20, -20, -20, -20, -20, -10],
               [20,   20,  -5, -10, -10,  -5,  20,  20],
               [15,   30,  10, -10,  -5,   0,  25,  15]]

king_scores_endgame = [[-40, -30, -25, -20, -20, -25, -30, -40],
                       [-30, -20, -10,   0,   0, -10, -20, -30],
                       [-30, -10,  20,  30,  30,  20, -10, -30],
                       [-30, -10,  30,  40,  40,  30, -10, -30],
                       [-30, -10,  30,  40,  40,  30, -10, -30],
                       [-30, -10,  20,  30,  30,  20, -10, -30],
                       [-30, -30,   0,   0,   0,   0, -30, -30],
                       [-40, -30, -25, -20, -20, -25, -30, -40]]

pawn_scores = [[0,   0,   0,   0,   0,  0,  0,  0],
               [50, 50,  50,  50,  50, 50, 50, 50],
               [10, 10,  20,  30,  30, 20, 10, 10],
               [5,   5,  10,  25,  25, 10,  5,  5],
               [0,   0,   0,  20,  20,  0, -5,  0],
               [5,  -5,   0,   5,   5,  0, -5,  5],
               [5,  10,  10, -30, -30, 10, 10,  5],
               [0,   0,   0,   0,   0,  0,  0,  0]]


piece_pos_scores = {"wR": rook_scores,
                    "bR": rook_scores[::-1],
                    "wN": knight_scores,
                    "bN": knight_scores[::-1],
                    "wB": bishop_scores,
                    "bB": bishop_scores[::-1],
                    "wQ": queen_scores,
                    "bQ": queen_scores[::-1],
                    "wK": king_scores,
                    "bK": king_scores[::-1],
                    "wp": pawn_scores,
                    "bp": pawn_scores[::-1],
                    }


def findRandomMove(validMoves):
    if len(validMoves) != 0:
        return validMoves[random.randint(0, len(validMoves)-1)]
    else:
        return None


# Helper method to make the first recursive call
def findBestMove(bs, validMoves):
    OpeningDatabase = {
        "": "e2e4",
        "b1c3": "d7d5",
        "c2c4": "e7e5",
        "e2e4": "e7e5",
        "d2d4": "d7d5",
        "g1f3": "d7d5",
        "e2e4c7c5": "g1f3",
        "e2e4c7c6": "b1c3",
        "e2e4d7d5": "e4d5",
        "e2e4e7e5": "b1c3",
        "e2e4e7e6": "d2d4",
        "d2d4d7d5c1f4": "c7c5",     # London Opening
        "d2d4d7d5c2c4": "e7e6",     # Queen's Gambit
        "d2d4d7d5g1f3": "g8f6",
        "e2e4e7e5b1c3": "b8c6",     # Vienna Game
        "e2e4e7e5d2d4": "e5d4",     # Center Game
        "e2e4e7e5f1c4": "b8c6",     # Bishop's Opening
        "e2e4e7e5g1f3": "b8c6",
        "e2e4e7e5f2f4": "d7d5",     # King's Gambit
        "d2d4d7d5c1f4c7c5c2c3": "b8c6",
        "d2d4d7d5c1f4c7c5e2e3": "b8c6",
        "d2d4d7d5c2c4e7e6b1c3": "g8f6",
        "d2d4d7d5c2c4e7e6c4d5": "e6d5",
        "d2d4d7d5c2c4e7e6g1f3": "g8f6",
        "e2e4e7e5b1c3b8c6f1c4": "g8f6",
        "e2e4e7e5f1c4b8c6b1c3": "g8f6",
        "e2e4e7e5g1f3b8c6b1c3": "g8f6",     # Two Knights Opening
        "e2e4e7e5g1f3b8c6d2d4": "e5d4",     # Scotch Game
        "e2e4e7e5g1f3b8c6f1b5": "a7a6",     # Spanish Opening
        "e2e4e7e5g1f3b8c6f1c4": "f8c5",     # Italian Game
        "d2d4d7d5c2c4e7e6b1c3g8f6c1g5": "f8b4",
        "d2d4d7d5c2c4e7e6g1f3g8f6c1g5": "f8b4",
        "e2e4e7e5b1c3b8c6f1c4g8f6d2d3": "f8b4",
        "e2e4e7e5g1f3b8c6b1c3g8f6f1b5": "f8b4",     # Four Knights Opening: Spanish Variation
        "e2e4e7e5g1f3b8c6b1c3g8f6f1c4": "f8b4",     # Four Knights Opening: Italian Variation
        "e2e4e7e5g1f3b8c6d2d4e5d4f3d4": "f8c5",
        "e2e4e7e5g1f3b8c6d2d4e5d4f1c4": "f8b4",     # Scotch Gambit
        "e2e4e7e5g1f3b8c6f1b5a7a6b5c6": "d7c6",     # Spanish Opening, Morphy Variation
        "e2e4e7e5g1f3b8c6f1b5a7a6b5a4": "g8f6",     # Spanish Opening, Morphy Variation
        "e2e4e7e5g1f3b8c6f1c4f8c5d2d3": "g8f6",     # Italian Game, Modern Variation
        "e2e4e7e5g1f3b8c6f1c4f8c5c2c3": "g8f6",     # Italian Game, Classical Variation
        "e2e4e7e5g1f3b8c6f1c4f8c5e1g1": "g8f6",
    }
    global nextMove
    nextMove = None
    if len(bs.moveLog) <= 7:
        playedMoves = getLogNotation(bs)
        dictMove = OpeningDatabase.get(playedMoves)
        if dictMove is not None:
            nextMove = ChessEngine.Move((ChessEngine.Move.ranksToRows.get(dictMove[1]), ChessEngine.Move.filesToCols.get(dictMove[0])), (ChessEngine.Move.ranksToRows.get(dictMove[3]), ChessEngine.Move.filesToCols.get(dictMove[2])), bs.board)
            p.time.wait(random.randint(200, 400))
    if nextMove is None:
        # random.shuffle(validMoves)
        NegaMaxAlphaBeta(bs, validMoves, DEPTH, 1 if bs.whiteToMove else -1, -CHECKMATE, CHECKMATE)
    return nextMove


def NegaMaxAlphaBeta(bs, validMoves, depth, turnMultiplier, alpha, beta):
    global nextMove
    if depth == 0 or len(validMoves) == 0:
        if bs.checkmate:
            if bs.whiteToMove:
                return turnMultiplier * -CHECKMATE-(depth*10)   # Black wins
            else:
                return turnMultiplier * CHECKMATE+(depth*10)    # White wins
        elif bs.stalemate:
            return STALEMATE
        # captureMoves = bs.getValidCaptures()
        # return QuiescenceSearch(bs, captureMoves, 2, 1 if bs.whiteToMove else -1, -CHECKMATE, CHECKMATE)
        return turnMultiplier * scoreBoard(bs)

    maxScore = -CHECKMATE*10
    for move in validMoves:
        bs.makeMove(move)
        nextMoves = bs.getValidMoves()
        score = -NegaMaxAlphaBeta(bs, nextMoves, depth - 1, -turnMultiplier, -beta, -alpha)
        if score > maxScore:
            maxScore = score
            if depth == DEPTH:
                nextMove = move
                print(move.getChessNotation(), score)
        bs.undoMove()
        if maxScore > alpha:
            alpha = maxScore
        if alpha >= beta:
            break
    return maxScore


# Returns a score, positive signals an advantage for White
def scoreBoard(bs):
    score = 0
    material_count = countMaterial(bs.board)
    for row in range(len(bs.board)):
        for col in range(len(bs.board[row])):
            square = bs.board[row][col]
            if square != "--":
                if square[1] != "K":
                    piece_pos_score = piece_pos_scores[square][row][col]
                else:   # King
                    if material_count <= 3000:  # Checks whether the endgame has been reached
                        piece_pos_score = king_scores_endgame[row][col]
                    else:
                        piece_pos_score = piece_pos_scores[square][row][col]
                if square[0] == "w":
                    score += piece_score[square[1]] + piece_pos_score
                elif square[0] == "b":
                    score -= piece_score[square[1]] + piece_pos_score
    return score


def countMaterial(board):
    materialCount = 0
    for row in board:
        for square in row:
            if square != "--":
                materialCount += piece_score[square[1]]
    return materialCount


def getLogNotation(bs):
    moves = ""
    if len(bs.moveLog) > 0:
        for i in range(0, len(bs.moveLog)):
            move = bs.moveLog[i].getChessNotation()
            moves = moves + move
    return moves

