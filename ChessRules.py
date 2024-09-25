"""
This is responsible for storing all the information about the current state of the chess game.
It is responsible for determining any valid move in any given board_state.
It is responsible for keeping a move log.
"""


class BoardState_and_Rules:

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
        self.columnCount = 8
        self.rowCount = 8
        self.swapDictionary = {
        'wR': 'bR', 'wN': 'bN', 'wB': 'bB', 'wQ': 'bQ', 'wK': 'bK', 'wp': 'bp',
        'bR': 'wR', 'bN': 'wN', 'bB': 'wB', 'bQ': 'wQ', 'bK': 'wK', 'bp': 'wp',
        '--': '--'
    }

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

    def makeMoveAZ(self, state, move, whiteKingLocation, blackKingLocation, enPassantLogs, currentCastlingRights, castleRightsLog):
        board = state
        if board[move.startRow][move.startCol] != "--":
            board[move.startRow][move.startCol] = "--"
            board[move.endRow][move.endCol] = move.pieceMoved

            if move.pieceMoved == 'wK':
                whiteKingLocation = (move.endRow, move.endCol)
            if move.pieceMoved == 'bK':
                blackKingLocation = (move.endRow, move.endCol)

            # Pawn Promotion
            if move.isPawnPromotion:
                board[move.endRow][move.endCol] = move.pieceMoved[0] + "Q"

            # En Passant
            if move.isEnpassantMove:
                board[move.startRow][move.endCol] = '--'  # Capturing the Piece
            if move.pieceMoved[1] == 'p' and abs(move.endRow - move.startRow) == 2:
                isPossibleEnpassant = ((move.startRow + move.endRow) // 2, move.endCol)
            else:
                isPossibleEnpassant = ()
            enPassantLogs.append(isPossibleEnpassant)

            # Castle Move
            if move.isCastleMove:
                if move.endCol > move.startCol:  # Kingside Castle
                    board[move.endRow][move.endCol - 1] = board[move.endRow][move.endCol + 1]  # Rook move
                    board[move.endRow][7] = "--"  # Removes Rook from original tile
                else:  # Queenside Castle
                    board[move.endRow][move.endCol + 1] = board[move.endRow][move.endCol - 2]
                    board[move.endRow][0] = "--"
            # Update Castling Rights
            currentCastlingRights = self.updateCastlingRights(move, state, currentCastlingRights)
            newCastleRights = CastleRights(currentCastlingRights.wKs, currentCastlingRights.wQs,
                                           currentCastlingRights.bKs, currentCastlingRights.bQs)
            castleRightsLog.append(newCastleRights)
            return board, blackKingLocation, whiteKingLocation, enPassantLogs, castleRightsLog


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
            self. currentCastlingRights = self.updateCastlingRights(move, board=self.board, castling_rights=self.currentCastlingRights)
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

    def updateCastlingRights(self, move, board, castling_rights):
        # Update castling rights based on the piece that was moved
        if move.pieceMoved == 'wK':
            castling_rights['wQs'] = False
            castling_rights['wKs'] = False
        elif move.pieceMoved == 'bK':
            castling_rights['bQs'] = False
            castling_rights['bKs'] = False

        # If a Rook is ever captured or moves
        if board[0][0] != "bR":
            castling_rights['bQs'] = False
        if board[0][7] != "bR":
            castling_rights['bKs'] = False
        if board[7][0] != "wR":
            castling_rights['wQs'] = False
        if board[7][7] != "wR":
            castling_rights['wKs'] = False

        return castling_rights

    def getValidMovesAZ(self, board, castlingRights, player, whiteKingLocation, blackKingLocation):
        temp_castlingRights = castlingRights
        moves = []
        inCheck, pins, checks = self.checkForPinsAndChecks(player, board, whiteKingLocation, blackKingLocation)
        if player == 1:
            kingRow = self.whiteKingLocation[0]
            kingCol = self.whiteKingLocation[1]
        else:
            kingRow = self.blackKingLocation[0]
            kingCol = self.blackKingLocation[1]

        if inCheck:
            if len(checks) == 1:  # 1 check: check can be blocked or King can be moved
                moves = self.getAllPossibleMoves(board, player, pins, whiteKingLocation, blackKingLocation)
                check = checks[0]
                checkRow, checkCol = check[0], check[1]
                pieceChecking = board[checkRow][checkCol]  # enemy piece causing the check
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
                self.getKingMoves(kingRow, kingCol, moves, board, whiteKingLocation, blackKingLocation)

        else:
            moves = self.getAllPossibleMoves(board, player, pins, whiteKingLocation, blackKingLocation)

        if len(moves) == 0:
            if inCheck:
                checkmate = True
            else:
                stalemate = True
        else:
            stalemate = False
            checkmate = False
        currentCastlingRights = temp_castlingRights

        self.getCastleMoves(kingRow, kingCol, moves, board, player, inCheck, currentCastlingRights)
        return moves

    def getValidMoves(self):
        temp_castlingRights = self.currentCastlingRights
        moves = []
        player = self.getPlayer()

        self.inCheck, self.pins, self.checks = self.checkForPinsAndChecks(player, self.board, self.whiteKingLocation, self.blackKingLocation)
        if self.whiteToMove:
            kingRow = self.whiteKingLocation[0]
            kingCol = self.whiteKingLocation[1]
        else:
            kingRow = self.blackKingLocation[0]
            kingCol = self.blackKingLocation[1]

        if self.inCheck:
            if len(self.checks) == 1:  # 1 check: check can be blocked or King can be moved
                moves = self.getAllPossibleMoves(self.board, player, self.pins, self.whiteKingLocation, self.blackKingLocation)
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
                self.getKingMoves(kingRow, kingCol, moves, self.board, self.whiteKingLocation, self.blackKingLocation)

        else:
            moves = self.getAllPossibleMoves(self.board, player, self.pins,self.whiteKingLocation, self.blackKingLocation)

        if len(moves) == 0:
            if self.inCheck:
                self.checkmate = True
            else:
                self.stalemate = True
        else:
            self.stalemate = False
            self.checkmate = False
        self.currentCastlingRights = temp_castlingRights

        self.getCastleMoves(kingRow, kingCol, moves, self.board, player, self.inCheck, self.currentCastlingRights)
        return moves

    def checkForPinsAndChecks(self, board, player, whiteKingLocation, blackKingLocation):
        pins = []  # tile where allied pinned piece is and the direction of pin
        checks = []  # tile where enemy is attacking
        inCheck = False
        if player == 1:
            enemyColor = 'b'
            allyColor = 'w'
            startRow = whiteKingLocation[0]
            startCol = whiteKingLocation[1]
        else:
            enemyColor = 'w'
            allyColor = 'b'
            startRow = blackKingLocation[0]
            startCol = blackKingLocation[1]
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for j in range(len(directions)):
            d = directions[j]
            possiblePins = ()  # reset possible pins
            for i in range(1, 8):  # stands for number of tiles away
                endRow = startRow + (d[0] * i)
                endCol = startCol + (d[1] * i)
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = board[endRow][endCol]
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
                endPiece = board[endRow][endCol]
                if endPiece[0] == enemyColor and endPiece[1] == 'N':  # enemy Knight attacking King
                    inCheck = True
                    checks.append((endRow, endCol, m[0], m[1]))
        return inCheck, pins, checks

    def getAllPossibleMoves(self, board, player, pins, whiteKingLocation, blackKingLocation):
        moves = []
        for r in range(len(board)):
            for c in range(len(board[r])):
                turn = board[r][c][0]
                if turn == "w" and player == 1 or turn == "b" and not player == 1:
                    piece = board[r][c][1]
                    if piece == "p":
                        self.getPawnMoves(r, c, moves, board, player, pins)
                    elif piece == "R":
                        self.getRookMoves(r, c, moves, board, player, pins)
                    elif piece == "N":
                        self.getKnightMoves(r, c, moves, board, player, pins)
                    elif piece == "B":
                        self.getBishopMoves(r, c, moves, board, player, pins)
                    elif piece == "Q":
                        self.getQueenMoves(r, c, moves, board, player, pins)
                    elif piece == "K":
                        self.getKingMoves(r, c, moves, board, whiteKingLocation, blackKingLocation)
        return moves

    def getPawnMoves(self, r, c, moves, board, player, pins):
        piecePinned = False
        pinDirection = ()
        for i in range(len(pins)-1, -1, -1):
            if pins[i][0] == r and pins[i][1] == c:
                piecePinned = True
                pinDirection = (pins[i][2], pins[i][3])
                pins.remove(pins[i])
                break

        if player == 1:
            moveDirection = -1
            startRow = 6
            enemyColor = 'b'
        else:
            moveDirection = 1
            startRow = 1
            enemyColor = 'w'

        # pawn advance
        if board[r + moveDirection][c] == '--':  # 1 square pawn advance
            if not piecePinned or pinDirection == (moveDirection, 0):
                moves.append(Move((r, c), (r + moveDirection, c), board))
                if r == startRow and board[r + (2 * moveDirection)][c] == '--':  # 2 square pawn advance
                    moves.append(Move((r, c), (r + (2 * moveDirection), c), board))
        # captures to left
        if c-1 >= 0:
            if not piecePinned or pinDirection == (moveDirection, -1):
                if board[r + moveDirection][c - 1][0] == enemyColor:
                    moves.insert(0, Move((r, c), (r + moveDirection, c - 1), board))
                if (r + moveDirection, c - 1) == self.isPossibleEnpassant:
                    moves.insert(0, Move((r, c), (r + moveDirection, c - 1), board, isEnpassantMove=True))
        # capture to right
        if c+1 <= 7:
            if not piecePinned or pinDirection == (moveDirection, 1):
                if board[r + moveDirection][c + 1][0] == enemyColor:
                    moves.insert(0, Move((r, c), (r + moveDirection, c + 1), board))
                if (r + moveDirection, c + 1) == self.isPossibleEnpassant:
                    moves.insert(0, Move((r, c), (r + moveDirection, c + 1), board, isEnpassantMove=True))

    def getRookMoves(self, r, c, moves, board, player, pins):
        piecePinned = False
        pinDirection = ()
        for i in range(len(pins) - 1, -1, -1):
            if pins[i][0] == r and pins[i][1] == c:
                piecePinned = True
                pinDirection = (pins[i][2], pins[i][3])
                if board[r][c][1] != 'Q':  # if we remove pin in case of a Queen then it will allow Bishop type moves on Queen.
                    pins.remove(pins[i])
                break
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up down left right
        enemyColor = 'b' if player == 1 else 'w'  # opponent's color according to current turn
        for d in directions:
            for i in range(1, 8):
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    if not piecePinned or pinDirection == d or pinDirection == (
                            -d[0], -d[1]):  # move towards the pic or away from it
                        if board[endRow][endCol] == '--':  # Empty square
                            moves.append(Move((r, c), (endRow, endCol), board))
                        elif board[endRow][endCol][0] == enemyColor:  # capture opponent's piece
                            moves.insert(0, Move((r, c), (endRow, endCol), board))
                            break
                        else:
                            break  # same color piece
                else:
                    break  # off board

    def getKnightMoves(self, r, c, moves, board, player, pins):
        piecePinned = False
        for i in range(len(pins) - 1, -1, -1):
            if pins[i][0] == r and pins[i][1] == c:
                piecePinned = True
                pins.remove(pins[i])
                break

        if not piecePinned:  # if a knight is pinned we can't move it anywhere
            directions = ((-1, -2), (-2, -1), (1, -2), (2, -1), (1, 2), (2, 1), (-1, 2), (-2, 1))
            allyColor = 'w' if player == 1 else 'b'  # opponent's color according to current turn
            for d in directions:
                endRow = r + d[0]
                endCol = c + d[1]
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    endPiece = board[endRow][endCol]
                    if endPiece[0] == "-":
                        moves.append(Move((r, c), (endRow, endCol), board))
                    elif endPiece[0] != allyColor:
                        moves.insert(0, Move((r, c), (endRow, endCol), board))

    def getBishopMoves(self, r, c, moves, board, player, pins):
        piecePinned = False
        pinDirection = ()
        for i in range(len(pins) - 1, -1, -1):
            if pins[i][0] == r and pins[i][1] == c:
                piecePinned = True
                pinDirection = (pins[i][2], pins[i][3])
                pins.remove(pins[i])
                break

        directions = ((-1, -1), (-1, 1), (1, -1), (1, 1))  # (top left) (top right) (bottom left) (bottom right)
        enemyColor = 'b' if player == 1 else 'w'  # opponent's color according to current turn
        for d in directions:
            for i in range(1, 8):
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                    if not piecePinned or pinDirection == d or pinDirection == (-d[0], -d[1]):
                        if board[endRow][endCol] == '--':  # Empty Square
                            moves.append(Move((r, c), (endRow, endCol), board))
                        elif board[endRow][endCol][0] == enemyColor:  # capture opponent's piece
                            moves.insert(0, Move((r, c), (endRow, endCol), board))
                            break
                        else:
                            break  # same color piece
                else:
                    break  # off board

    def getQueenMoves(self, r, c, moves, board, player, pins):
        self.getRookMoves(r, c, moves, board, player, pins)
        self.getBishopMoves(r, c, moves, board, player, pins)

    def getKingMoves(self, r, c, moves, board, whiteKingLocation, blackKingLocation):
        player = self.getPlayer()
        directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
        allyColor = "w" if player == 1 else "b"
        for d in directions:
            endRow = r + d[0]
            endCol = c + d[1]
            if 0 <= endRow <= 7 and 0 <= endCol <= 7:
                endPiece = board[endRow][endCol]
                if endPiece[0] != allyColor:
                    # temporarily move the king to the new location
                    if allyColor == "w":
                        whiteKingLocation = (endRow, endCol)
                    if allyColor == "b":
                        blackKingLocation = (endRow, endCol)
                    inCheck, pins, checks = self.checkForPinsAndChecks(board, player, whiteKingLocation, blackKingLocation)
                    if not inCheck:
                        moves.append(Move((r, c), (endRow, endCol), board))

                    # place king back
                    if allyColor == "w":
                        whiteKingLocation = (r, c)
                    if allyColor == "b":
                        blackKingLocation = (r, c)

    def getPlayer(self):
        if self.whiteToMove:
            player = 1
        else:
            player = -1
        return player

    '''
    Gets the list of all of the king's castling move -> for the king at(r,c);
    '''

    def getCastleMoves(self, r, c, moves, board, player, inCheck, currentCastlingRights):
        if inCheck:
            return
        if (player == 1 and currentCastlingRights.wKs) or \
                (not player == 1 and currentCastlingRights.bKs):
            self.getKingSideCastleMoves(r, c, moves, board, player)

        if (player == 1 and currentCastlingRights.wQs) or \
                (not player == 1 and currentCastlingRights.bQs):
            self.getQueenSideCastleMoves(r, c, moves, board, player)

    def getKingSideCastleMoves(self, r, c, moves, board, player):
        if board[r][c + 1] == '--' and board[r][c + 2] == '--':
            if not self.tileUnderAttack(r, c + 1, player, board) and not self.tileUnderAttack(r, c + 2, player, board):
                moves.append(Move((r, c), (r, c + 2), board, isCastleMove=True))

    def getQueenSideCastleMoves(self, r, c, moves, board, player):
        if board[r][c - 1] == '--' and board[r][c - 2] == '--' and board[r][c - 3] == '--':
            if not self.tileUnderAttack(r, c - 1, player, board) and not self.tileUnderAttack(r, c - 2, player, board):
                moves.append(Move((r, c), (r, c - 2), board, isCastleMove=True))

    '''
   Checks if sq (r,c) is under attack or not
   '''

    def tileUnderAttack(self, r, c, player, board):
        # LESS OPTIMISED WAY -> generating opponent's moves and then seeing -> but not a lot as max 16 opponent's piece and max 8 moves per piece -> total 128 moves -> much less for computer to calculate

        # MORE OPTIMISED ALGORITHM -> only 64 + 8 -> 72 calculations
        # here we move away from the square we want to calculate threat from
        allyColor = 'w' if player == 1 else 'b'
        enemyColor = 'b' if player == 1 else 'w'
        directions = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        underAttack = False
        for j in range(len(directions)):  # stands for direction => [0,3] -> orthogonal || [4,7] -> diagonal
            d = directions[j]
            for i in range(1, 8):  # stands for number of tiles away
                endRow = r + (d[0] * i)
                endCol = c + (d[1] * i)
                if 0 <= endRow < 8 and 0 <= endCol < 8:
                    endPiece = board[endRow][endCol]
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
                endPiece = board[endRow][endCol]
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


def getLogNotation(bs):
    moves = ""
    if len(bs.moveLog) > 0:
        for i in range(0, len(bs.moveLog)):
            move = bs.moveLog[i].getChessNotation()
            moves = moves + move
    return moves