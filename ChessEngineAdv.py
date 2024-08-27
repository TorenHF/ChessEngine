"""
This is responsible for storing all the information about the current state of the chess game.
It is responsible for determining any valid move in any given board_state.
It is responsible for keeping a move log.
"""


class BoardState:

    def __init__(self):
        # The board is 8x8. It is represented by a 2d list
        # Each element of the list is described by two characters in accordance to algebraic notation:
        # The first character represents the color of the piece:
        # "b" (black), "w" (white)
        # The second character represents the type of the piece:
        # "R" (Rook), "N" (Knight), "B" (Bishop), "Q" (Queen), "K" (King), "p" (pawn)
        # "--" represents an empty tile

        self.board = [['bR', 'bN', 'bB', 'bQ', 'bK', 'bB', 'bN', 'bR'],
                      ['bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp', 'bp'],
                      ['--', '--', '--', '--', '--', '--', '--', '--'],
                      ['--', '--', '--', '--', '--', '--', '--', '--'],
                      ['--', '--', '--', '--', '--', '--', '--', '--'],
                      ['--', '--', '--', '--', '--', '--', '--', '--'],
                      ['wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp', 'wp'],
                      ['wR', 'wN', 'wB', 'wQ', 'wK', 'wB', 'wN', 'wR']]

        self.whiteToMove = True
        self.moveLog = []
        # Keeping track of kings' placement to make legal move generation and castling more simple
        self.whiteKingLocation = (7, 4)
        self.blackKingLocation = (0, 4)

        self.inCheck = False
        self.pins = []
        self.checks = []
        self.checkmate = False
        self.stalemate = False

        # En passant
        self.isPossibleEnpassant = ()  # Tile where en passant capture can happen
        self.enPassantLogs = [self.isPossibleEnpassant]

        # Castling
        self.currentCastlingRights = CastleRights(True, True, True, True)
        self.castleRightsLog = [CastleRights(self.currentCastlingRights.wKs, self.currentCastlingRights.wQs,
                                             self.currentCastlingRights.bKs, self.currentCastlingRights.bQs)]

    def makeMove(self, move):
        if self.board[move.startRow][move.startCol] != "--":
            self.board[move.startRow][move.startCol] = "--"
            self.board[move.endRow][move.endCol] = move.pieceMoved
            self.moveLog.append(move)

            if move.pieceMoved == 'wK':
                self.whiteKingLocation = (move.endRow, move.endCol)
            if move.pieceMoved == 'bK':
                self.blackKingLocation = (move.endRow, move.endCol)

            # Pawn Promotion
            if move.isPawnPromotion:
                self.board[move.endRow][move.endCol] = move.pieceMoved[0] + "Q"

            # En Passant
            if move.isEnpassantMove:
                self.board[move.startRow][move.endCol] = '--'  # Capturing the Piece
            if move.pieceMoved[1] == 'p' and abs(move.endRow - move.startRow) == 2:
                self.isPossibleEnpassant = ((move.startRow + move.endRow) // 2, move.endCol)
            else:
                self.isPossibleEnpassant = ()
            self.enPassantLogs.append(self.isPossibleEnpassant)

            # Castle Move
            if move.isCastleMove:
                if move.endCol > move.startCol:    # Kingside Castle
                    self.board[move.endRow][move.endCol-1] = self.board[move.endRow][move.endCol+1]   # Rook move
                    self.board[move.endRow][7] = "--"   # Removes Rook from original tile
                else:   # Queenside Castle
                    self.board[move.endRow][move.endCol+1] = self.board[move.endRow][move.endCol-2]
                    self.board[move.endRow][0] = "--"
            # Update Castling Rights
            self.updateCastlingRights(move)
            newCastleRights = CastleRights(self.currentCastlingRights.wKs, self.currentCastlingRights.wQs,
                                           self.currentCastlingRights.bKs, self.currentCastlingRights.bQs)
            self.castleRightsLog.append(newCastleRights)

            self.whiteToMove = not self.whiteToMove  # swap the turn

    def undoMove(self):
        if len(self.moveLog) == 0:
            return
        move = self.moveLog.pop()
        self.board[move.startRow][move.startCol] = move.pieceMoved
        self.board[move.endRow][move.endCol] = move.pieceCaptured
        self.whiteToMove = not self.whiteToMove
        # Update the King's placement if needed
        if move.pieceMoved == 'wK':
            self.whiteKingLocation = (move.startRow, move.startCol)
        if move.pieceMoved == 'bK':
            self.blackKingLocation = (move.startRow, move.startCol)
        # Undo Enpassant Move
        if move.isEnpassantMove:
            self.board[move.endRow][move.endCol] = '--'  # Removes the pawn that captured En passant
            self.board[move.startRow][move.endCol] = move.pieceCaptured  # Places back the pawn that was captured
        self.enPassantLogs.pop()
        self.isPossibleEnpassant = self.enPassantLogs[-1]
        # Undo castle rights
        self.castleRightsLog.pop()
        self.currentCastlingRights.wKs = self.castleRightsLog[-1].wKs
        self.currentCastlingRights.wQs = self.castleRightsLog[-1].wQs
        self.currentCastlingRights.bKs = self.castleRightsLog[-1].bKs
        self.currentCastlingRights.bQs = self.castleRightsLog[-1].bQs
        # Undo castle move
        if move.isCastleMove:
            if move.endCol > move.startCol:    # Kingside Castle
                self.board[move.endRow][move.endCol+1] = self.board[move.endRow][move.endCol-1]
                self.board[move.endRow][5] = "--"
            else:   # Queenside Castle
                self.board[move.endRow][move.endCol-2] = self.board[move.endRow][move.endCol+1]
                self.board[move.endRow][3] = "--"

        self.checkmate = False
        self.stalemate = False

    def updateCastlingRights(self, move):
        if move.pieceMoved == 'wK':
            self.currentCastlingRights.wQs = False
            self.currentCastlingRights.wKs = False
        elif move.pieceMoved == 'bK':
            self.currentCastlingRights.bQs = False
            self.currentCastlingRights.bKs = False
        # If a Rook is ever captured or moves
        if self.board[0][0] != "bR":
            self.currentCastlingRights.bQs = False
        if self.board[0][7] != "bR":
            self.currentCastlingRights.bKs = False
        if self.board[7][0] != "wR":
            self.currentCastlingRights.wQs = False
        if self.board[7][7] != "wR":
            self.currentCastlingRights.wKs = False

    def getValidMoves(self):
        temp_castlingRights = self.currentCastlingRights
        moves = []
        self.inCheck, self.pins, self.checks = self.checkForPinsAndChecks()
        if self.whiteToMove:
            kingRow = self.whiteKingLocation[0]
            kingCol = self.whiteKingLocation[1]
        else:
            kingRow = self.blackKingLocation[0]
            kingCol = self.blackKingLocation[1]

        if self.inCheck:
            if len(self.checks) == 1:  # 1 check: check can be blocked or King can be moved
                moves = self.getAllPossibleMoves()
                check = self.checks[0]
                checkRow, checkCol = check[0], check[1]
                pieceChecking = self.board[checkRow][checkCol]  # enemy piece causing the check
                validTiles = []  # tiles on which pieces can be interposed and block the check
                if pieceChecking[1] == 'N':
                    validTiles.append((checkRow, checkCol))
                else:
                    for i in range(1, 8):
                        validTile = (kingRow + check[2] * i, kingCol + check[3] * i)
                        validTiles.append(validTile)
                        if validTile[0] == checkRow and validTile[1] == checkCol:  # Capture of checking piece
                            break
                for i in range(len(moves)-1, -1, -1):
                    if moves[i].pieceMoved[1] != 'K':
                        if not (moves[i].endRow, moves[i].endCol) in validTiles:
                            moves.remove(moves[i])

            else:  # double check: King must move
                self.getKingMoves(kingRow, kingCol, moves)

        else:
            moves = self.getAllPossibleMoves()

        if len(moves) == 0:
            if self.inCheck:
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.stalemate = False
            self.checkmate = False
        self.currentCastlingRights = temp_castlingRights

        self.getCastleMoves(kingRow, kingCol, moves)
        return moves

    def checkForPinsAndChecks(self):
        pins = []  # tile where allied pinned piece is and the direction of pin
        checks = []  # tile where enemy is attacking
        inCheck = False
        if self.whiteToMove:
            enemyColor = 'b'
            allyColor = 'w'
            startRow = self.whiteKingLocation[0]
            startCol = self.whiteKingLocation[1]
        else:
            enemyColor = 'w'
            allyColor = 'b'
            startRow = self.blackKingLocation[0]
            startCol = self.blackKingLocation[1]
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for j in range(len(directions)):
            d = directions[j]
            possiblePins = ()  # reset possible pins
            for i in range(1, 8):  # stands for number of tiles away
                endRow = startRow + (d[0] * i)
                endCol = startCol + (d[1] * i)
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece[0] == allyColor and endPiece[1] != 'K':
                        if possiblePins == ():  # 1st piece that too ally -> might be a pin
                            possiblePins = (endRow, endCol, d[0], d[1])
                        else:  # 2nd ally piece -> no pins or checks in this direction
                            break
                    elif endPiece[0] == enemyColor:
                        pieceType = endPiece[1]
                        # Five different possibilities here:
                        # 1) orthogonally away Rook
                        # 2) Diagonally away Bishop
                        # 3) 1 sq. diagonally away Pawn
                        # 4) Any Direction away Queen
                        # 5) 1 sq. any direction King
                        if (0 <= j <= 3 and pieceType == 'R') or \
                                (4 <= j <= 7 and pieceType == 'B') or \
                                (i == 1 and pieceType == 'p') and (
                                (enemyColor == 'w' and 6 <= j <= 7) or (enemyColor == 'b' and 4 <= j <= 5)) or \
                                (pieceType == 'Q') or \
                                (i == 1 and pieceType == 'K'):
                            if possiblePins == ():  # no piece interposed -->check
                                inCheck = True
                                checks.append((endRow, endCol, d[0], d[1]))
                                break
                            else:  # possible pin
                                pins.append(possiblePins)
                                break
                        else:  # enemy piece not applying check
                            break
                else:  # OFF BOARD
                    break
        # Knight Checks:
        knightMoves = [(-1, -2), (-2, -1), (1, -2), (2, -1), (1, 2), (2, 1), (-1, 2), (-2, 1)]
        for m in knightMoves:
            endRow = startRow + m[0]
            endCol = startCol + m[1]
            if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] == enemyColor and endPiece[1] == 'N':  # enemy Knight attacking King
                    inCheck = True
                    checks.append((endRow, endCol, m[0], m[1]))
        return inCheck, pins, checks

    def getAllPossibleMoves(self):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if turn == "w" and self.whiteToMove or turn == "b" and not self.whiteToMove:
                    piece = self.board[r][c][1]
                    if piece == "p":
                        self.getPawnMoves(r, c, moves)
                    elif piece == "R":
                        self.getRookMoves(r, c, moves)
                    elif piece == "N":
                        self.getKnightMoves(r, c, moves)
                    elif piece == "B":
                        self.getBishopMoves(r, c, moves)
                    elif piece == "Q":
                        self.getQueenMoves(r, c, moves)
                    elif piece == "K":
                        self.getKingMoves(r, c, moves)
        return moves

    def getPawnMoves(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins)-1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        if self.whiteToMove:
            moveDirection = -1
            startRow = 6
            enemyColor = 'b'
        else:
            moveDirection = 1
            startRow = 1
            enemyColor = 'w'

        # pawn advance
        if self.board[r + moveDirection][c] == '--':  # 1 square pawn advance
            if not piecePinned or pinDirection == (moveDirection, 0):
                moves.append(Move((r, c), (r + moveDirection, c), self.board))
                if r == startRow and self.board[r + (2 * moveDirection)][c] == '--':  # 2 square pawn advance
                    moves.append(Move((r, c), (r + (2 * moveDirection), c), self.board))
        # captures to left
        if c-1 >= 0:
            if not piecePinned or pinDirection == (moveDirection, -1):
                if self.board[r + moveDirection][c - 1][0] == enemyColor:
                    moves.insert(0, Move((r, c), (r + moveDirection, c - 1), self.board))
                if (r + moveDirection, c - 1) == self.isPossibleEnpassant:
                    moves.insert(0, Move((r, c), (r + moveDirection, c - 1), self.board, isEnpassantMove=True))
        # capture to right
        if c+1 <= 7:
            if not piecePinned or pinDirection == (moveDirection, 1):
                if self.board[r + moveDirection][c + 1][0] == enemyColor:
                    moves.insert(0, Move((r, c), (r + moveDirection, c + 1), self.board))
                if (r + moveDirection, c + 1) == self.isPossibleEnpassant:
                    moves.insert(0, Move((r, c), (r + moveDirection, c + 1), self.board, isEnpassantMove=True))

    def getRookMoves(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                if self.board[r][c][1] != 'Q':  # if we remove pin in case of a Queen then it will allow Bishop type moves on Queen.
                    self.pins.remove(self.pins[i])
                break
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up down left right
        enemyColor = 'b' if self.whiteToMove else 'w'  # opponent's color according to current turn
        for d in directions:
            for i in range(1, 8):
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    if not piecePinned or pinDirection == d or pinDirection == (
                            -d[0], -d[1]):  # move towards the pic or away from it
                        if self.board[endRow][endCol] == '--':  # Empty square
                            moves.append(Move((r, c), (endRow, endCol), self.board))
                        elif self.board[endRow][endCol][0] == enemyColor:  # capture opponent's piece
                            moves.insert(0, Move((r, c), (endRow, endCol), self.board))
                            break
                        else:
                            break  # same color piece
                else:
                    break  # off board

    def getKnightMoves(self, r, c, moves):
        piecePinned = False
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                self.pins.remove(self.pins[i])
                break

        if not piecePinned:  # if a knight is pinned we can't move it any where
            directions = ((-1, -2), (-2, -1), (1, -2), (2, -1), (1, 2), (2, 1), (-1, 2), (-2, 1))
            allyColor = 'w' if self.whiteToMove else 'b'  # opponent's color according to current turn
            for d in directions:
                endRow = r + d[0]
                endCol = c + d[1]
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    endPiece = self.board[endRow][endCol]
                    if endPiece[0] == "-":
                        moves.append(Move((r, c), (endRow, endCol), self.board))
                    elif endPiece[0] != allyColor:
                        moves.insert(0, Move((r, c), (endRow, endCol), self.board))

    def getBishopMoves(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # (top left) (top right) (bottom left) (bottom right)
        enemyColor = 'b' if self.whiteToMove else 'w'  # opponent's color according to current turn
        for d in directions:
            for i in range(1, 8):
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0], -d[1]):
                        if self.board[endRow][endCol] == '--':  # Empty Square
                            moves.append(Move((r, c), (endRow, endCol), self.board))
                        elif self.board[endRow][endCol][0] == enemyColor:  # capture opponent's piece
                            moves.insert(0, Move((r, c), (endRow, endCol), self.board))
                            break
                        else:
                            break  # same color piece
                else:
                    break  # off board

    def getQueenMoves(self, r, c, moves):
        self.getRookMoves(r, c, moves)
        self.getBishopMoves(r, c, moves)

    def getKingMoves(self, r, c, moves):
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        allyColor = "w" if self.whiteToMove else "b"
        for d in directions:
            endRow = r + d[0]
            endCol = c + d[1]
            if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] != allyColor:
                    # temporarily move the king to the new location
                    if allyColor == "w":
                        self.whiteKingLocation = (endRow, endCol)
                    if allyColor == "b":
                        self.blackKingLocation = (endRow, endCol)
                    inCheck, pins, checks = self.checkForPinsAndChecks()
                    if not inCheck:
                        moves.append(Move((r, c), (endRow, endCol), self.board))

                    # place king back
                    if allyColor == "w":
                        self.whiteKingLocation = (r, c)
                    if allyColor == "b":
                        self.blackKingLocation = (r, c)

    # Attempt at creating functions that only return valid capture moves to further improve minimax search depth
    def getValidCaptures(self):
        tempCastlingRights = self.currentCastlingRights
        moves = []
        self.inCheck, self.pins, self.checks = self.checkForPinsAndChecks()
        if self.whiteToMove:
            kingRow = self.whiteKingLocation[0]
            kingCol = self.whiteKingLocation[1]
        else:
            kingRow = self.blackKingLocation[0]
            kingCol = self.blackKingLocation[1]

        if self.inCheck:
            if len(self.checks) == 1:  # only 1 check -> block check or move king
                moves = self.getAllPossibleCaptures()
                # to block check -> move piece btw King and enemy piece Attacking
                check = self.checks[0]
                checkRow = check[0]
                checkCol = check[1]
                pieceChecking = self.board[checkRow][checkCol]  # enemy piece causing the check
                validSquares = []  # sq. to which we can bring our piece to block the check
                # if Night then it must be captured or move the king
                if pieceChecking[1] == 'N':
                    validSquares.append((checkRow, checkCol))
                # other pieces can be blocked
                else:
                    for i in range(1, 8):
                        validSquare = (kingRow + check[2] * i, kingCol + check[3] * i)
                        validSquares.append(validSquare)
                        if validSquare[0] == checkRow and validSquare[
                            1] == checkCol:  # once you reach attacking piece stop.
                            break

                # get rid of any move that doesn't block king or capture the piece
                for i in range(len(moves) - 1, -1, -1):
                    if moves[i].pieceMoved[1] != 'K':  # move doesn't move King so must block or Capture
                        if not (moves[i].endRow, moves[i].endCol) in validSquares:
                            moves.remove(moves[i])

            else:  # double check -> must move king
                self.getKingMoves(kingRow, kingCol, moves)

        else:
            moves = self.getAllPossibleCaptures()

        if len(moves) == 0:
            if self.inCheck:
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.stalemate = False
            self.checkmate = False
        self.currentCastlingRights = tempCastlingRights

        self.getCastleMoves(kingRow, kingCol, moves)
        return moves

    def getAllPossibleCaptures(self):
        moves = []
        for r in range(len(self.board)):
            for c in range(len(self.board[r])):
                turn = self.board[r][c][0]
                if turn == "w" and self.whiteToMove or turn == "b" and not self.whiteToMove:
                    piece = self.board[r][c][1]
                    if piece == "p":
                        self.getPawnCaptures(r, c, moves)
                    elif piece == "R":
                        self.getRookCaptures(r, c, moves)
                    elif piece == "N":
                        self.getKnightCaptures(r, c, moves)
                    elif piece == "B":
                        self.getBishopCaptures(r, c, moves)
                    elif piece == "Q":
                        self.getQueenCaptures(r, c, moves)
                    elif piece == "K":
                        self.getKingCaptures(r, c, moves)
        return moves

    def getPawnCaptures(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        if self.whiteToMove:
            moveAmount = -1
            enemyColor = 'b'
        else:
            moveAmount = 1
            enemyColor = 'w'

        # captures to left
        if c - 1 >= 0:
            if not piecePinned or pinDirection == (moveAmount, -1):
                if self.board[r + moveAmount][c - 1][0] == enemyColor:
                    moves.insert(0, Move((r, c), (r + moveAmount, c - 1), self.board))
                if (r + moveAmount, c - 1) == self.isPossibleEnpassant:
                    moves.insert(0, Move((r, c), (r + moveAmount, c - 1), self.board, isEnpassantMove=True))

        # capture to right
        if c + 1 < len(self.board):
            if not piecePinned or pinDirection == (moveAmount, 1):
                if self.board[r + moveAmount][c + 1][0] == enemyColor:
                    moves.insert(0, Move((r, c), (r + moveAmount, c + 1), self.board))
                if (r + moveAmount, c + 1) == self.isPossibleEnpassant:
                    moves.insert(0, Move((r, c), (r + moveAmount, c + 1), self.board, isEnpassantMove=True))

    def getRookCaptures(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                if self.board[r][c][
                    1] != 'Q':  # if we remove pin in case of a Queen then it will allow Bishop type moves on Queen.
                    self.pins.remove(self.pins[i])
                break
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up down left right
        enemyColor = 'b' if self.whiteToMove else 'w'  # opponent's color according to current turn
        for d in directions:
            for i in range(1, 8):
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow < len(self.board) and 0 <= endCol < len(self.board[endRow]):
                    if not piecePinned or pinDirection == d or pinDirection == (
                            -d[0], -d[1]):  # move towards the pic or away from it
                        if self.board[endRow][endCol][0] == enemyColor:  # capture opponent's piece
                            moves.insert(0, Move((r, c), (endRow, endCol), self.board))
                            break
                        else:
                            break  # same color piece
                else:
                    break  # off board

    def getKnightCaptures(self, r, c, moves):
        piecePinned = False
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                self.pins.remove(self.pins[i])
                break

        if not piecePinned:  # if a knight is pinned we can't move it any where
            directions = ((-1, -2), (-2, -1), (1, -2), (2, -1), (1, 2), (2, 1), (-1, 2), (-2, 1))
            allyColor = 'w' if self.whiteToMove else 'b'  # opponent's color according to current turn
            for d in directions:
                endRow = r + d[0]
                endCol = c + d[1]
                if endRow >= 0 and endRow < len(self.board) and endCol >= 0 and endCol < len(self.board[endRow]):
                    endPiece = self.board[endRow][endCol]
                    if endPiece[0] != allyColor:
                        moves.insert(0, Move((r, c), (endRow, endCol), self.board))

    def getBishopCaptures(self, r, c, moves):
        piecePinned = False
        pinDirection = ()
        for i in range(len(self.pins) - 1, -1, -1):
            if self.pins[i][0] == r and self.pins[i][1] == c:
                piecePinned = True
                pinDirection = (self.pins[i][2], self.pins[i][3])
                self.pins.remove(self.pins[i])
                break

        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # (top left) (top right) (bottom left) (bottom right)
        enemyColor = 'b' if self.whiteToMove else 'w'  # opponent's color according to current turn
        for d in directions:
            for i in range(1, 8):
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if endRow >= 0 and endRow < len(self.board) and endCol >= 0 and endCol < len(self.board[endRow]):
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0], -d[1]):
                        if self.board[endRow][endCol][0] == enemyColor:  # capture opponent's piece
                            moves.insert(0, Move((r, c), (endRow, endCol), self.board))
                            break
                        else:
                            break  # same color piece
                else:
                    break  # off board

    def getQueenCaptures(self, r, c, moves):
        self.getRookCaptures(r, c, moves)
        self.getBishopCaptures(r, c, moves)

    def getKingCaptures(self, r, c, moves):
        pass

    '''
    Gets the list of all of the king's castling move -> for the king at(r,c);
    '''

    def getCastleMoves(self, r, c, moves):
        if self.inCheck:
            return
        if (self.whiteToMove and self.currentCastlingRights.wKs) or \
                (not self.whiteToMove and self.currentCastlingRights.bKs):
            self.getKingSideCastleMoves(r, c, moves)

        if (self.whiteToMove and self.currentCastlingRights.wQs) or \
                (not self.whiteToMove and self.currentCastlingRights.bQs):
            self.getQueenSideCastleMoves(r, c, moves)

    def getKingSideCastleMoves(self, r, c, moves):
        if self.board[r][c + 1] == '--' and self.board[r][c + 2] == '--':
            if not self.tileUnderAttack(r, c + 1) and not self.tileUnderAttack(r, c + 2):
                moves.append(Move((r, c), (r, c + 2), self.board, isCastleMove=True))

    def getQueenSideCastleMoves(self, r, c, moves):
        if self.board[r][c - 1] == '--' and self.board[r][c - 2] == '--' and self.board[r][c - 3] == '--':
            if not self.tileUnderAttack(r, c - 1) and not self.tileUnderAttack(r, c - 2):
                moves.append(Move((r, c), (r, c - 2), self.board, isCastleMove=True))

    '''
   Checks if sq (r,c) is under attack or not
   '''

    def tileUnderAttack(self, r, c):
        # LESS OPTIMISED WAY -> generating opponent's moves and then seeing -> but not a lot as max 16 opponent's piece and max 8 moves per piece -> total 128 moves -> very less for computer to calculate

        # MORE OPTIMISED ALGORITHM -> only 64 + 8 -> 72 calculations
        # here we move away from the square we want to calculate threat from
        allyColor = 'w' if self.whiteToMove else 'b'
        enemyColor = 'b' if self.whiteToMove else 'w'
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        underAttack = False
        for j in range(len(directions)):  # stands for direction => [0,3] -> orthogonal || [4,7] -> diagonal
            d = directions[j]
            for i in range(1, 8):  # stands for number of tiles away
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = self.board[endRow][endCol]
                    if endPiece[0] == allyColor:
                        break
                    elif endPiece[0] == enemyColor:
                        pieceType = endPiece[1]
                        # Five different possibilities here:
                        # 1) orthogonally away Rook
                        # 2) Diagonally away Bishop
                        # 3) 1 sq. diagonally away Pawn
                        # 4) Any Direction away Queen
                        # 5) 1 sq. any direction King
                        if (0 <= j <= 3 and pieceType == 'R') or \
                                (4 <= j <= 7 and pieceType == 'B') or \
                                (i == 1 and pieceType == 'p') and (
                                (enemyColor == 'w' and 6 <= j <= 7) or (enemyColor == 'b' and 4 <= j <= 5)) or \
                                (pieceType == 'Q') or \
                                (i == 1 and pieceType == 'K'):

                            return True
                        else:  # enemy piece not applying check
                            break
                else:  # OFF BOARD
                    break
        if underAttack:
            return True
        # CHECK FOR KNIGHT CHECKS:
        knightMoves = [(-1, -2), (-2, -1), (1, -2), (2, -1), (1, 2), (2, 1), (-1, 2), (-2, 1)]
        for m in knightMoves:
            endRow = r + m[0]
            endCol = c + m[1]
            if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                endPiece = self.board[endRow][endCol]
                if endPiece[0] == enemyColor and endPiece[1] == 'N':  # enemy knight attacking king
                    return True
        return False


class CastleRights:

    def __init__(self, wKs, wQs, bKs, bQs):
        self.wKs = wKs
        self.wQs = wQs
        self.bKs = bKs
        self.bQs = bQs


class Move:

    ranksToRows = {"1": 7, "2": 6, "3": 5, "4": 4,
                   "5": 3, "6": 2, "7": 1, "8": 0}
    rowsToRanks = {v: k for k, v in ranksToRows.items()}
    filesToCols = {"a": 0, "b": 1, "c": 2, "d": 3,
                   "e": 4, "f": 5, "g": 6, "h": 7}
    colsToFiles = {v: k for k, v in filesToCols.items()}

    def __init__(self, startTile, endTile, board, isEnpassantMove=False, isCastleMove=False):
        self.startRow = startTile[0]
        self.startCol = startTile[1]
        self.endRow = endTile[0]
        self.endCol = endTile[1]
        self.pieceMoved = board[self.startRow][self.startCol]
        self.pieceCaptured = board[self.endRow][self.endCol]
        self.isPawnPromotion = (self.pieceMoved == "wp" and self.endRow == 0 or self.pieceMoved == "bp" and self.endRow == 7)
        self.isEnpassantMove = isEnpassantMove
        if isEnpassantMove:
            self.pieceCaptured = "wp" if self.pieceMoved == "bp" else "bp"
        self.isCastleMove = isCastleMove
        self.moveId = self.startRow * 1000 + self.startCol * 100 + self.endRow * 10 + self.endCol

    def getChessNotation(self):
        return self.getFileRank(self.startRow, self.startCol) + self.getFileRank(self.endRow, self.endCol)

    def getFileRank(self, r, c):
        return self.colsToFiles[c] + self.rowsToRanks[r]

    def __eq__(self, other):
        return isinstance(other, Move) and self.moveId == other.moveId
