# -*- coding: utf-8 -*-

import random
from copy import copy
import time

import numpy as np

from agent import Agent
from random_agent import RandomAgent
from stingy_agent import StingyAgent
from generous_agent import GenerousAgent
from util import *

SILENT = True

class ClassicMonteCarlo(Agent):
    """An agent that """

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        self.game_tree =[]
        self.sim_time = kwargs.get('simul_time',1)
        
    def get_action(self, state, legal_moves):
        if not legal_moves:
            return None
        
        move = self.classic_monte_carlo_search(state = state,
                                               playout_agent_white = RandomAgent(self.reversi,-1),
                                               playout_agent_black = RandomAgent(self.reversi,1),
                                               decision_time = self.sim_time
                                               )      
#        print state[0],move
        return move   
        
    def classic_monte_carlo_search(self,state,
                                playout_agent_white,
                                playout_agent_black,                                
                                decision_time):

        new_reversi = copy(self.reversi)
        
        root_node = OthelloGameNode(state)
        game_node = root_node
        self.update_game_tree_node(game_node) 
               
        start_time = time.time()   
        while time.time()-start_time < decision_time:
            # keep on selecting child nodes as long as they are elligable
                  
            while self.is_selectable(game_node,self.game_tree):  
                game_state = self.select(game_node,self.game_tree)
                game_node = OthelloGameNode(game_state)
                self.update_game_tree_node(game_node) 
            
            if new_reversi.winner(game_node.state):
                result = self.reversi.winner(game_node.state)
                self.backpropagate(result)

            else:             
                # expand search space by looking at a new child node       
                game_node_child_move = self.expand(game_node.state)
                
                game_node_child_state = self.reversi.next_state(game_node.state, game_node_child_move)
                game_node_child = OthelloGameNode(game_node_child_state)
                game_node.add_child(game_node_child)

                [self.update_game_tree_node(gn) for gn in [game_node,game_node_child]]      

                # simulate a game based on the specified agent
                result = self.simulate(game_node_child_state,
                                  playout_agent_white,
                                  playout_agent_black)   
                
                # upgrade node values
                self.backpropagate(result)
             
            # go through the tree again
            game_node = root_node
            
            self.update_game_tree_node(game_node) 

        best_next_state = self.select(root_node,self.game_tree)

        for move in self.reversi.legal_moves(state):
            possible_state = self.reversi.next_state(state, move)
            if possible_state == best_next_state:
                return move

    def reset(self):
        pass

    def observe_win(self, winner):
        pass
         
    def update_game_tree_node(self,node):
        
        def mark(node):
            node.seen = True
            return node

        node.seen = True
        states_in_game_tree = [nd.state for nd in self.game_tree]   

        if node.state not in states_in_game_tree:     
            self.game_tree.append(node)

        else:
            self.game_tree = [mark(nd) if node.state == nd.state else nd
                              for nd in self.game_tree ] 

        return
    
    def find_state_in_game_tree(self,query_state):
                         
        return [node for node in self.game_tree if node.state == query_state ][0]  
        
    def backpropagate(self,result):

        def update_node_on_result(node,result):
            node.t +=1
            if node.seen == True:
                node.wins += result
                node.visited += 1
                node.seen = False 
                node.val_func = self.value_function(node)
                node.white_perc = 1.0*node.wins/node.visited
            return node
                
        self.game_tree = [update_node_on_result(node,result) for node in self.game_tree]
             
        return
    
    def simulate(self,state,playout_agent_white,playout_agent_black):
        new_reversi = copy(self.reversi)
        new_reversi.game_state = state
        new_reversi.white_agent = playout_agent_white
        new_reversi.black_agent = playout_agent_black
        result, white_score, black_score = new_reversi.play_game()
         
        return 0.5 * (result + 1)
               
    def expand(self, state):

        legal_moves = self.reversi.legal_moves(state)
        if legal_moves:
            return random.choice(legal_moves)
        else:
            False
        
    def select(self,node,game_tree):  
        
        node_in_game_tree = self.find_state_in_game_tree(node.state)
        # TODO node_in_game tree is different from a simple node instance, 
        
#        children = node_in_game_tree.children_nodes
        children = node.children_nodes
        
        color = node_in_game_tree.state[1]
        if color == 1:
            best_node = sorted(children,key=lambda x: x.val_func, 
                               reverse=True)[0]
        else:
            best_node = sorted(children,key=lambda x: x.val_func, 
                               reverse=True)[-1]
        
        return best_node.state
      
    def is_selectable(self,node,game_tree):
        
        node_in_game_tree = self.find_state_in_game_tree(node.state)

        if not node.children_nodes:
            return False    
        
        if len(node.children_nodes) == 1:
            return False  
            
        child_value_list = [child.val_func for child in node.children_nodes]

        # decision if max or min is best depends on if player is white or black
        color = node_in_game_tree.state[1]
        child_value_list.sort(reverse=False) if color ==1 else child_value_list.sort(reverse=True)
        
        if child_value_list[0] == child_value_list[1]:
            return False
        else:
            return True
            
    def value_function(self,node):
        
        wi = node.wins
        ni = node.visited
        sum_ni = node.t
        c = np.sqrt(2)
        vf = 1.0 * wi/ni + 1.0 * c * np.sqrt(1.0 * sum_ni/ni)  

        return vf
        
        
class OthelloGameNode():
    """"""
    
    def __init__(self,state):
        self.wins = 0
        self.visited = 0
        self.t = 0
        self.seen = False
        self.state = state
        self.children_nodes =[]
        self.final_node = False
        self.val_func = 0
        self.white_wins = 0.
           
    def mark_as_seen(self):
        self.seen = True
        return
        
    def add_child(self,node):
        found_children_nodes = [ch_n for ch_n in self.children_nodes if node.state == ch_n.state ]
        if not found_children_nodes:
            self.children_nodes.append(node)
        return
        
if __name__ =="__main__":
    pass