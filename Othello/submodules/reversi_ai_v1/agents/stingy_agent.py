# -*- coding: utf-8 -*-

import random
from copy import copy
from agents.agent import Agent
from util import *

class StingyAgent(Agent):
    """An agent that simply chooses the move that 
    gets him the most stones possible"""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        
    def get_action(self, state, legal_moves):
        if not legal_moves:
            return None
            
        move = self.stingy_search(state)
        return move 
        
    def stingy_search(self,state):
        
        legal_moves = self.reversi.legal_moves(state)
        
        option_value_list =[]
        for move in legal_moves:
            new_reversi = copy(self.reversi)
            possible_state = new_reversi.next_state(state, move)
  
            black_count, white_count = possible_state[0].get_stone_counts()
            if self.color == -1:
                option_value_list.append((black_count,move))
            else:
                option_value_list.append((white_count,move))

        best_move = sorted(option_value_list,key=lambda x: x[0], reverse=True)[0]

        return best_move[1]


    def reset(self):
        pass

    def observe_win(self, winner):
        pass
