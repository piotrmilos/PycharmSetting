from __future__ import absolute_import
import sys
sys.path.insert(0,'..')

from itertools import product

from reversi_ai.game.board import Board 
from reversi_ai.game.reversi import Reversi
from reversi_ai.util import *
 
        
def stone_other(color):
   if color == BLACK:
       return WHITE
   elif color == WHITE:
       return BLACK
   else:
       raise("Wrong color passed")

       
class GymBoard(Board):    
    
    def play(self,move,color):
        x,y = move
        self.board[y][x] = color
        if color == WHITE:
            self.white_stones += 1
        elif color == BLACK:
            self.black_stones += 1
            
    def coord_to_ij(self,coordinate):
        y = int(1.0*coordinate/self.size)
        x = coordinate % self.size
        return x,y
        
    def ij_to_coord(self,x,y):
        return 1.0*y*self.size+x
        
    def get_legal_coords(self,color):
        board = self.board
        
        if board.is_full():
            return []

        board_size = board.get_size()
        moves = []  # list of x,y positions valid for color

        for x,y in product(range(board_size), range(board_size)):
            if self.is_valid_move(board,color, x, y):
                moves.append((x, y))

    @staticmethod
    def is_in_bounds(x, y, size):
        return 0 <= x < size and 0 <= y < size
                
    @staticmethod
    def is_valid_move(board,color, x, y):

        piece = board.board[y][x]
        if piece != EMPTY:
            return False

        enemy = opponent[color]

        # now check in all directions, including diagonal
        for dy,dx in product(range(-1, 2),range(-1, 2)):
            if dy == 0 and dx == 0:
                continue

            # there needs to be >= 1 opponent piece
            # in this given direction, followed by 1 of player's piece
            distance = 1
            yp = (distance * dy) + y
            xp = (distance * dx) + x

            while is_in_bounds(xp, yp, board.size) and board.board[yp][xp] == enemy:
                distance += 1
                yp = (distance * dy) + y
                xp = (distance * dx) + x

            if distance > 1 and is_in_bounds(xp, yp, board.size) and board.board[yp][xp] == color:
                return True
        return False
        
