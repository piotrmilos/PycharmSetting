# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import random
from copy import copy
import time

import numpy as np

from game.board import Board, BLACK, WHITE, EMPTY
from game.reversi import Reversi
from agents.agent import Agent
from agents.random_agent import RandomAgent
from agents.stingy_agent import StingyAgent
from agents.generous_agent import GenerousAgent
from util import *

SILENT = True

class ClassicMonteCarlo(Agent):
    """An agent that """

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        
    def get_action(self, state, legal_moves):
        if not legal_moves:
            return None
        
        move = self.classic_monte_carlo_search(state = state,
                                               playout_agent_white = RandomAgent(self.reversi,1),
                                               playout_agent_black = RandomAgent(self.reversi,-1),
                                               decision_time = 5
                                               )        
        return move   
        
    def classic_monte_carlo_search(self,state,
                                playout_agent_white,
                                playout_agent_black,                                
                                decision_time):

        GameState = list(known_nodes)
        root_node
        
        while there is time:
            next_known_node = root_node
            
            while next_known_node is not last_known_node:
                GameState.select_known_node
                next_known_node = GameState.select.last_node

            new_node = last_known_node.expand
            new_node.simulate_result
            GameState.backprop_update_on_all_nodes_involved
                
        return GameState.best_move
          
    def reset(self):
        pass

    def observe_win(self, winner):
        pass

    def backpropagate(self,result):
    
    
    def simulate(self,playout_agent_white,playout_agent_black):

               
    def expand(self, state,legal_moves,

        
    def select(self):     

        
    def value_function(self):

        
                
class OthelloGameNode():
    """"""
    
    def __init__(self,state):

           
    def mark_as_seen(self):

        
    def is_selectable(self):


    def add_child(self,node):


class OthelloGameSpace():
    """"""

    def __init__(self):
        self.game_nodes =[]

    def add_node(self,node):   


    def find_node(self,state):    


           

