# -*- coding: utf-8 -*-

import pygame
from pygame import *
import os, sys
from pygame.locals import *

BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (0,128,0)

BLUE = (0,255,255)
RED = (255,0,0)

#0 white 1 black
PLAYERS = ['w','b']

ADJACENT = [ (-1,-1),(0,-1),(1,-1), (-1,0),(1,0), (-1,1),(0,1),(1,1) ]

BOARD_SIZE = 14
ROW = ['e']*BOARD_SIZE

ACTIVEPLAYER = 0

class Board():
    
    def __init__(self,board_size = BOARD_SIZE):
        self.grid = []
        self.board_size = board_size
        for i in range(self.board_size):
            self.grid += [ROW]

        self.board_length = 50*self.board_size
        self.board = pygame.Surface((self.board_length,self.board_length))
        self.board.fill( GREEN )
        
        initial_position_low = int(self.board_size/2. -1) 
        initial_position_high = int(self.board_size/2.) 
        print initial_position_high,initial_position_low
        
        self.setGridEntry(initial_position_low,initial_position_low,'w')
        self.setGridEntry(initial_position_high,initial_position_high,'w')
        self.setGridEntry(initial_position_low,initial_position_high,'b')
        self.setGridEntry(initial_position_high,initial_position_low,'b')
    
    def redraw(self):
        screen.blit(self.board, (0,0) )
        for i in range(0,self.board_length,50):
            pygame.draw.line(self.board, WHITE, (0,i), (self.board_length,i), 1)
            pygame.draw.line(self.board, WHITE, (i,0), (i,self.board_length), 1)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.getGridEntry(i,j) == 'w':
                    pygame.draw.circle(self.board, WHITE, ( i*50 + 25, j*50 + 25 ), 10, 0 )
                if self.getGridEntry(i,j) == 'b':
                    pygame.draw.circle(self.board, BLACK, ( i*50 + 25, j*50 + 25 ), 10, 0 )
                if self.getGridEntry(i,j) == 'c':
                    pygame.draw.circle(self.board, BLUE, ( i*50 - 25, j*50 - 25 ), 10, 0 )
            
    def getGridEntry(self, y, x):
        if ( y in range(self.board_size) and x in range(self.board_size) ):
            return self.grid[x][y]
        else:
            return 'x'

    def setGridEntry(self, y, x, entry):
        self.grid[x] = self.grid[x][:y] + [entry] + self.grid[x][y+1:]
    
    #returns list of (x,y) locations of valid moves for player
    def findAvailableMoves(self, p):
        candidateMoves = []
        finalMovesList = []
        movesString = PLAYERS[ 1-p ] + PLAYERS[p]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if ( self.getGridEntry( i, j ) == PLAYERS[1-p] ):
                    for (x1,y1) in ADJACENT:
                        if ( self.getGridEntry( i + x1, j + y1 ) == 'e' ):
                            candidateMoves += [ ( i + x1, j + y1 ) ]
    
        for pos in candidateMoves:
            (x, y) = pos
            halfRow = ['']*self.board_size
            shortList = []
        
            for i in range( len(ADJACENT) ):
                (Ax, Ay) = ADJACENT[i]
                if (myBoard.getGridEntry(x+Ax,y+Ay) == PLAYERS[1-p] ):
                    shortList += [(Ax, Ay)]
            for i in range(self.board_size):
                for j in range( len(shortList) ):
                    (Ax, Ay) = shortList[j]
                    halfRow[j] += self.getGridEntry(x+Ax*i, y+Ay*i)
                for r in halfRow:
                    if movesString in r:
                        finalMovesList += [ (x, y) ]
        return finalMovesList


class Game():
    
    def __init__(self,board_size = BOARD_SIZE):
        self.board_size = board_size
        self.board_length = 50*board_size
        self.movesList = []
        self.X = 0
        self.Y = 0
        self.activePlayer = 0
        self.pieceCount = [2,2]
        self.debug = 'off'
    
    #Makes move assuming valid move is selected
    def makeMove(self, x, y):
        if ( (self.X, self.Y) not in self.movesList ):
            return
        takenPieces = []
        candidateRow = [ [], [], [], [], [], [], [], [] ]
        shortList = []
        
        for i in range( len(ADJACENT) ):
            (Ax, Ay) = ADJACENT[i]
            if (myBoard.getGridEntry(self.X+Ax,self.Y+Ay) == PLAYERS[1-self.activePlayer] ):
                shortList += [(Ax, Ay)]
        halfRow = ['']*self.board_size
        for i in range(self.board_size):
            for j in range( len(shortList) ):
                (Ax, Ay) = shortList[j]
                if ( 0 <= self.X + Ax*i < self.board_size and 0 <= self.Y + Ay*i < self.board_size ):
                    if ( myBoard.getGridEntry(self.X + Ax*i,self.Y + Ay*i) == PLAYERS[1-self.activePlayer] ):
                        candidateRow[j] += [(self.X + Ax*i,self.Y + Ay*i)]
                    elif ( myBoard.getGridEntry(self.X + Ax*i,self.Y + Ay*i) == PLAYERS[self.activePlayer] ):
                        takenPieces += candidateRow[j]
        takenPieces = list( set(takenPieces) )
        for piece in takenPieces:
            (x, y) = piece
            myBoard.setGridEntry(x, y, PLAYERS[self.activePlayer] )
            self.pieceCount[self.activePlayer] += 1
            self.pieceCount[1-self.activePlayer] -= 1
        myBoard.setGridEntry(self.X,self.Y, PLAYERS[self.activePlayer])
        self.pieceCount[self.activePlayer] += 1
        #Other player's turn now
        self.activePlayer = ( 1- self.activePlayer )

    def EventHandler(self, key):
        if key == K_ESCAPE:
            os.sys.exit(0)
        else:
            (x,y) = pygame.mouse.get_pos()
            self.makeMove( x,y )
            
    def updateMouse(self):
        self.movesList = myBoard.findAvailableMoves(self.activePlayer)
        pos = pygame.mouse.get_pos()
        (x, y) = pos
        self.X = (x/50)
        self.Y = (y/50)
        if (x > self.board_length or y > self.board_length):
            textline = "Othello (c) 2012 Dan Allen"
        else:
            if self.debug == 'on':
                textline = str( self.X ) + ", " + str( self.Y ) + "X:" + str(x) + "Y:" + str(y) + "P:" + PLAYERS[self.activePlayer]
            else:
                textline = "P:" + PLAYERS[self.activePlayer]
        
            boxX = ( ( x / 50 ) * 50 ) + 1
            boxY = ( ( y / 50 ) * 50 ) + 1
            if ( (self.X, self.Y) in self.movesList ):
                pygame.draw.polygon(screen, BLUE, ( (boxX, boxY), (boxX+47,boxY), (boxX+47, boxY+47), (boxX, boxY+47) ), 2)
            else:
                pygame.draw.polygon(screen, RED, ( (boxX, boxY), (boxX+47,boxY), (boxX+47, boxY+47), (boxX, boxY+47) ), 2)
            
        textbox = pygame.Surface( (500,100) )
        if pygame.font:
            font = pygame.font.Font(None, 36)
            text = font.render(textline, 1, (255, 255, 255))
            textbox.blit(text, (0,0) )
        textpos = (0,self.board_length)
        screen.blit(textbox, textpos)
        
        scorebox = pygame.Surface( (500,100) )
        textline = "White: " + str(self.pieceCount[0]) + " Black: " + str(self.pieceCount[1])
        if pygame.font:
            font = pygame.font.Font(None, 36)
            text = font.render(textline, 1, (255, 255, 255))
            scorebox.blit(text, (0,0) )
        textpos = (0,450)
        screen.blit(scorebox, textpos)
        

myBoard = Board()
myGame = Game()

clock = pygame.time.Clock()

screen = pygame.display.set_mode((1000, 1000))

def main():
    pygame.init()
    while True:
        clock.tick(60)
        for e in event.get():
            if e.type == pygame.QUIT: #if EXIT clicked
                os.sys.exit(0) #close cleanly
            if e.type == pygame.KEYDOWN:
                myGame.EventHandler(e)
            if e.type == pygame.MOUSEBUTTONDOWN:
                myGame.EventHandler(e)
        myBoard.redraw()
        myGame.updateMouse()
        pygame.display.flip()
main()