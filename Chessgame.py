# Tutorial used created by Eddie Sharick: https://www.youtube.com/watch?v=EnYui0e73Rs
# Similar projects: https://github.com/mikolaj-skrzypczak/chess-engine, https://github.com/mnahinkhan/Chess


"""
This is the main driver file.
It is responsible for handling the user input and displaying the current BoardState object.
"""
import numpy as np
import pygame as p
import ChessRules as ChessEngine
# from Chess import ChessEngine # Weaker version of the program with more naive generation of legal moves
import BestMoveFinder


BOARD_WIDTH = BOARD_HEIGHT = 600
MOVE_LOG_PANEL_WIDTH = 300
MOVE_LOG_PANEL_HEIGHT = BOARD_HEIGHT
Dimension = 8
# A chessboard is a 8x8 grid
TileSize = BOARD_HEIGHT // Dimension
MaxFPS = 30
# This is relevant for the animations of the pieces
Images = {}
colors = [p.Color(215, 185, 105), p.Color(95, 60, 30)]


# This initializes a dictionary of the chess images. It is an expensive operation, therefore it is only called once.
def loadImages():
    pieces = ["bR", "bN", "bB", "bQ", "bK", "bp", "wR", "wN", "wB", "wQ", "wK", "wp"]
    for piece in pieces:
        Images[piece] = p.transform.scale(p.image.load(piece + ".png"), (TileSize, TileSize))
        # This is responsible for downloading images


# This is responsible for the title and the icon of the program
p.display.set_caption("Deep Black")
icon = p.image.load("mechanical-gears.png")
p.display.set_icon(icon)


# This is the main driver of the code. It is responsible for handling user input and uploading the graphics
def main():
    p.init()
    screen = p.display.set_mode((BOARD_WIDTH + MOVE_LOG_PANEL_WIDTH, BOARD_HEIGHT))
    clock = p.time.Clock()
    moveSound = p.mixer.Sound("move-self.mp3")
    screen.fill(p.Color("white"))
    moveLogFont = p.font.SysFont("Verdana", 20, False, False)
    bs = ChessEngine.BoardState_and_Rules()
    validMoves = bs.getValidMoves()
    move_count = 0
    moveMade = False    # Flag variable for when a move is made
    animate = False     # Flag variable for when an animation must be made
    loadImages()
    running = True
    gameOver = False
    playerOne = True
    playerTwo = False
    tileSelected = ()
    # This is responsible for storing the selected tile, it stores row, column
    playerClicks = []
    # This is responsible for keeping track of the player's clicks
    while running:
        humanTurn = (bs.whiteToMove and playerOne) or (not bs.whiteToMove and playerTwo)
        for e in p.event.get():
            if e.type == p.QUIT:
                running = False
            elif e.type == p.MOUSEBUTTONDOWN:
                if not gameOver and humanTurn:
                    location = p.mouse.get_pos()
                    # this is responsible for x,y location of the mouse
                    col = location[0]//TileSize
                    row = location[1]//TileSize
                    if tileSelected == (row, col) or col >= 8:
                        # This is responsible for checking if the player clicked the same square twice
                        tileSelected = ()       # deselects the clicked tile
                        playerClicks = []       # clears player's clicks
                    else:
                        tileSelected = (row, col)
                        playerClicks.append(tileSelected)       # Append for both the first and second clicks
                        if len(playerClicks) == 2:
                            move = ChessEngine.Move(playerClicks[0], playerClicks[1], bs.board)
                            for i in range(len(validMoves)):
                                if move == validMoves[i]:
                                    print(move.getChessNotation())
                                    bs.makeMove(bs, validMoves[i])
                                    moveMade = True
                                    animate = True
                                    tileSelected = ()       # resets the player's clicks
                                    playerClicks = []
                                    print(validMoves)
                            if not moveMade:
                                playerClicks = [tileSelected]

            elif e.type == p.KEYDOWN:
                if e.key == p.K_LEFT:      # undoes a move when the left arrow key is pressed
                    bs.undoMove()
                    moveMade = True
                    animate = False
                    gameOver = False
                    move_count -= 1
                elif e.key == p.K_r:
                    bs = ChessEngine.BoardState()
                    validMoves = bs.getValidMoves()
                    tileSelected = ()
                    playerClicks = []
                    moveMade = False
                    animate = False
                    gameOver = False
                    move_count = 0

        # AI move finder logic
        if not gameOver and not humanTurn:
            validMoves = bs.getValidMoves()
            print(validMoves)
            aiMove = BestMoveFinder.MCTS.findMove(bs, validMoves)
            bs.makeMove(aiMove)
            moveMade = True
            animate = True
        if moveMade:
            ChessEngine.getLogNotation(bs)
            moveSound.play()
            if animate:
                animateMove(bs.moveLog[-1], screen, bs.board, clock)
            validMoves = bs.getValidMoves()
            moveMade = False
            animate = False
            move_count += 1

        DrawBoardState(screen, bs, validMoves, tileSelected, moveLogFont)
        if bs.checkmate:
            gameOver = True
            if bs.whiteToMove:
                drawEndGameText(screen, "Black wins, Congratulations!")
            else:
                drawEndGameText(screen, "White wins, Congratulations!")
        elif bs.stalemate:
            drawEndGameText(screen, "Stalemate")
        clock.tick(MaxFPS)
        p.display.flip()


# This is responsible for all the graphics on the board
def DrawBoardState(screen, bs, validMoves, tileSelected, moveLogFont):
    DrawTiles(screen)
    # This is responsible for drawing the board tiles
    HighlightTiles(screen, bs, validMoves, tileSelected)
    if len(bs.moveLog) > 0:
        highlightLastMove(screen, bs.moveLog[-1])
    DrawPieces(screen, bs.board)
    # This is responsible for drawing pieces on top of the tiles
    drawMoveLog(screen, bs, moveLogFont)


# This is responsible for drawing the tiles of the board
def DrawTiles(screen):
    for r in range(Dimension):
        for c in range(Dimension):
            color = colors[((r+c) % 2)]
            p.draw.rect(screen, color, p.Rect(c*TileSize, r*TileSize, TileSize, TileSize))


# This is responsible for drawing the pieces on the board
def DrawPieces(screen, board):
    for r in range(Dimension):
        for c in range(Dimension):
            piece = board[r][c]
            if piece != "--":
                screen.blit(Images[piece], p.Rect(c * TileSize, r * TileSize, TileSize, TileSize))


# This is responsible for highlighting tiles when a move is to be made
def HighlightTiles(screen, bs, validMoves, tileSelected):
    if tileSelected != ():
        r, c = tileSelected
        if bs.board[r][c][0] == ("w" if bs.whiteToMove else "b"):   # Checks that the selected piece is an ally
            s = p.Surface((TileSize, TileSize))
            s.set_alpha(80)
            s.fill(p.Color("green"))
            screen.blit(s, (c*TileSize, r*TileSize))
            for move in validMoves:
                if move.startRow == r and move.startCol == c:
                    screen.blit(s, (move.endCol*TileSize, move.endRow*TileSize))


# This is responsible for highlighting the start and end tiles of the last move made
def highlightLastMove(screen, move):
    tile = p.Surface((TileSize, TileSize))
    tile.set_alpha(80)
    tile.fill(p.Color("yellow"))
    screen.blit(tile, (move.startCol * TileSize, move.startRow * TileSize))
    screen.blit(tile, (move.endCol * TileSize, move.endRow * TileSize))


def animateMove(move, screen, board, clock):
    global colors
    change_row = move.endRow - move.startRow
    change_col = move.endCol - move.startCol
    framesPerTile = 5
    frameCount = max(abs(change_row), abs(change_col)) * framesPerTile
    if change_row != 0 and change_col != 0 or (change_row == 1 or change_col == 1):
        frameCount += framesPerTile
    for frame in range(frameCount+1):
        r, c = (move.startRow + change_row*frame/frameCount, move.startCol + change_col*frame/frameCount)
        DrawTiles(screen)
        DrawPieces(screen, board)
        # Erase the moving piece in the ending tile
        color = colors[(move.endRow + move.endCol) % 2]
        endTile = p.Rect(move.endCol*TileSize, move.endRow*TileSize, TileSize, TileSize)
        p.draw.rect(screen, color, endTile)
        # Draw captured piece
        if move.pieceCaptured != "--":
            endTile = p.Rect(move.endCol * TileSize, move.endRow * TileSize, TileSize, TileSize)
            screen.blit(Images[move.pieceCaptured], endTile)
        # Draw moving piece
        screen.blit(Images[move.pieceMoved], p.Rect(c*TileSize, r*TileSize, TileSize, TileSize))
        p.display.flip()
        clock.tick(60)


def drawMoveLog(screen, bs, font):
    moveLogRect = p.Rect(BOARD_WIDTH, 0, MOVE_LOG_PANEL_WIDTH, MOVE_LOG_PANEL_HEIGHT)
    p.draw.rect(screen, p.Color("black"), moveLogRect)
    moveLog = bs.moveLog
    moveTexts = []
    for i in range(0, len(moveLog), 2):
        moveString = str(i//2 + 1) + ". " + moveLog[i].getChessNotation() + " "
        if i < len(moveLog) - 1:
            moveString += moveLog[i+1].getChessNotation()
        moveTexts.append(moveString)
    movesPerRow = 2
    padding = 5
    textY = padding
    for i in range(0, len(moveTexts), movesPerRow):
        text = ""
        for j in range(movesPerRow):
            if i + j < len(moveTexts):
                text += " " + moveTexts[i+j]
        textObject = font.render(text, True, p.Color("white"))
        textLocation = moveLogRect.move(padding, textY)
        screen.blit(textObject, textLocation)
        textY += textObject.get_height()


def drawEndGameText(screen, text):
    font = p.font.SysFont("Verdana", 40, True, False)
    textObject = font.render(text, False, p.Color("black"))
    textLocation = p.Rect(0, 0, BOARD_WIDTH, BOARD_HEIGHT).move(BOARD_WIDTH / 2 - textObject.get_width() / 2, BOARD_HEIGHT / 2 - textObject.get_height() / 2)
    screen.blit(textObject, textLocation)
    textObject = font.render(text, False, (180, 180, 180))
    screen.blit(textObject, textLocation.move(2, 2))


if __name__ == "__main__":
    main()
