import random
from copy import copy,deepcopy

from agent import Agent
from util import *

class AlphaBetaAgent(Agent):
    """"""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color
        self.game_tree = []

    def get_action(self, state, legal_moves):
        if not legal_moves:
            return None
        
        move = self.alpha_beta_search(state = state)
        return move   
        
    def alpha_beta_search(self,state):
        
        root_node = OthelloGameNode(state)
        root_node.node_type = "max"

        evaluated_children = self.evaluate_children(root_node,depth = 5)
        
        for minmax_val,ch_nd in evaluated_children:
            ch_nd.value = minmax_val
            root_node.add_child(ch_nd)
        
        return self.choose_best_move(root_node)
 
    def evaluate_children(self,node,depth):

        children = self.find_children(node)
        minamax_values = [self.minimax(ch_nd,depth) for ch_nd in children]  
        return zip(minamax_values,children)
                          
    def minimax(self,node,depth,v_min=-1.,v_max=+1.):
#        print "minimax",depth,self.value_function(node),node.node_type
        
        if self.is_final(node):
            if self.is_final(node):
                print "final",node.state[0],self.value_function(node)
            return self.value_function(node)
        
        if node.node_type == "max":
            v = v_min
            for ch_nd in self.find_children(node):
                v_new = self.minimax(ch_nd,depth-1.)
                if v_new > v:
                    v = v_new
                if v> v_max:
                    return v_max
            return v
                    
        if node.node_type == "min":
            v = v_max
            for ch_nd in self.find_children(node):
                v_new = self.minimax(ch_nd,depth-1.)
                if v_new < v:
                    v = v_new
                if v < v_min:
                    return v_min
            return v
         
    def is_final(self,node):
        True if self.reversi.winner(node.state) else False

    def find_children(self,node):
        legal_moves = self.reversi.legal_moves(node.state)
        child_states =[]
        for i,move in enumerate(legal_moves):

            child_states.append(self.reversi.next_state(node.state, move))      
      
        child_nodes = [OthelloGameNode(ch_st) for ch_st in child_states]
            
        def assign_node_type(child_node,parent_node):

            if parent_node.node_type == "max":
                child_node.node_type = "min"
            else:
                child_node.node_type = "max"
            return child_node
                
        child_nodes = [assign_node_type(ch_nd,node) for ch_nd in child_nodes]

        return child_nodes
        
    def choose_best_move(self,node):
        
        color = node.state[1]
        new_reversi = copy(self.reversi)
        if color == 1:
            best_node = sorted(node.children_nodes,key=lambda x: x.value, 
                               reverse=True)[0]
        else:
            best_node = sorted(node.children_nodes,key=lambda x: x.value, 
                               reverse=True)[-1]
        
        for move in new_reversi.legal_moves(node.state):
            possible_state = new_reversi.next_state(node.state, move)
            if possible_state == best_node.state:
                return move

    def value_function(self,node):
        if self.is_final(node):
            return self.reversi.winner(node.state)
        else:
            black_count, white_count = self.stone_counts(node.state)
            estimate = 1.0*(white_count - black_count) / (white_count + black_count)
            return estimate
            
    def stone_counts(self,state):
        black_count =  state[0].black_stones
        white_count =  state[0].white_stones
        return black_count, white_count
    
    def reset(self):
        pass

    def observe_win(self, winner):
        pass

class OthelloGameNode():
    """"""
    
    def __init__(self,state):
        self.value = False
        self.state = state
        self.node_type = False
        self.children_nodes =[]
        
    def add_child(self,node):
        found_children_nodes = [ch_n for ch_n in self.children_nodes if node.state == ch_n.state ]
        if not found_children_nodes:
            self.children_nodes.append(node)
        pass