# -*- coding: utf-8 -*-

import random
from copy import copy
import time

from game.board import Board, BLACK, WHITE, EMPTY
from agents.agent import Agent
from agents.random_agent import RandomAgent
from agents.stingy_agent import StingyAgent
from agents.generous_agent import GenerousAgent
from util import *

SILENT = True

class PureMonteCarlo(Agent):
    """An agent that """

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        
    def get_action(self, state, legal_moves):
        if not legal_moves:
            return None
        
        move = self.pure_monte_carlo_search(state,
                                            RandomAgent(self.reversi,1),
                                            RandomAgent(self.reversi,-1),
                                            100
                                            )        
        return move   
        
    def pure_monte_carlo_search(self,state,
                                playout_agent_white,
                                playout_agent_black,
                                playout_nr):
        
        legal_moves = self.reversi.legal_moves(state)
        color = state[1]

        playout_results =[]
        for move in legal_moves:
            start = time.time()
            playout_score = self.get_playout_score(state,
                                                   playout_agent_white,
                                                   playout_agent_black,
                                                   playout_nr)
            seconds_spent = time.time() - start
            if SILENT != True:
                print("Time per playout {}".format(seconds_spent*1000))
            playout_results.append((playout_score,move))
            
        if color == 1:
            best_move = sorted(playout_results,key=lambda x: x[0], 
                               reverse=True)[-1]
        else:
            best_move = sorted(playout_results,key=lambda x: x[0], 
                               reverse=True)[0]

#        self.reversi.print_board(state)
#        print(playout_results)
#        print("Best move: {}".format(best_move))

        return best_move[1]

    def get_playout_score(self,state,playout_agent_white,
                          playout_agent_black,playout_nr):
        
        result = 0
        for _ in range(playout_nr):
            new_reversi = copy(self.reversi)
            new_reversi.game_state = state
            new_reversi.white_agent = playout_agent_white
            new_reversi.black_agent = playout_agent_black
            
            winner, white_score, black_score = new_reversi.play_game()
#            print winner, white_score, black_score
            result += 1.0*(winner + 1.)/2. 
                
        return 1.0*result/playout_nr

    def reset(self):
        pass

    def observe_win(self, winner):
        pass

