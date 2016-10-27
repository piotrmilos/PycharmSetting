# -*- coding: utf-8 -*-

"""
Game of Hex
"""

from six import StringIO
import sys

from itertools import product

import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

def make_random_policy(np_random):
    def random_policy(state,player_color):
        possible_moves = OthelloEnv.get_possible_actions(state,player_color)

        # No moves left
        if len(possible_moves) == 0:
            return None
        a = np_random.randint(len(possible_moves))
        return possible_moves[a]
    return random_policy

class OthelloEnv(gym.Env):
    """
    Hex environment. Play against a fixed opponent.
    """
    BLACK = 0
    WHITE = 1
    metadata = {"render.modes": ["ansi","human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_move_mode, board_size):
        """
        Args:
            player_color: Stone color for the agent. Either 'black' or 'white'
            opponent: An opponent policy
            observation_type: State encoding
            illegal_move_mode: What to do when the agent makes an illegal move. Choices: 'raise' or 'lose'
            board_size: size of the Hex board
        """
        assert isinstance(board_size, int) and board_size >= 1, 'Invalid board size: {}'.format(board_size)
        self.board_size = board_size

        colormap = {
            'black': OthelloEnv.BLACK,
            'white': OthelloEnv.WHITE,
        }
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color must be 'black' or 'white', not {}".format(player_color))

        self.opponent = opponent
    
        assert observation_type in ['numpy3c']
        self.observation_type = observation_type

        assert illegal_move_mode in ['lose', 'raise']
        self.illegal_move_mode = illegal_move_mode

        if self.observation_type != 'numpy3c':
            raise error.Error('Unsupported observation type: {}'.format(self.observation_type))

        # One action for each board position and resign
        self.action_space = spaces.Discrete(self.board_size ** 2)
        observation = self.reset()
        self.observation_space = spaces.Box(np.zeros(observation.shape), np.ones(observation.shape))

        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

        # Update the random policy if needed
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            else:
                raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent

        return [seed]

    def _reset(self):
        
        self._seed()
        
        self.state = np.zeros((3, self.board_size, self.board_size))
        self.state[2,:,:] = 1.0
        
        center_positions = [int(self.board_size/2.-1),int(self.board_size/2.)]

        # set initial black stones
        self.state[0, center_positions[1],center_positions[0] ] = 1.0
        self.state[0, center_positions[0],center_positions[1]] = 1.0
        self.state[2, center_positions[1],center_positions[0] ] = 0.0
        self.state[2, center_positions[0],center_positions[1]] = 0.0
        # set initial white stones
        self.state[1, center_positions[0],center_positions[0]] = 1.0
        self.state[1, center_positions[1],center_positions[1]] = 1.0
        self.state[2, center_positions[0],center_positions[0] ] = 0.0
        self.state[2, center_positions[1],center_positions[1]] = 0.0

        self.to_play = OthelloEnv.BLACK
        self.done = False
        
        # Let the opponent play if it's not the agent's turn
        if self.player_color != self.to_play:
            a = self.opponent_policy(self.state,self.to_play)
            OthelloEnv.make_move(self.state, a, OthelloEnv.BLACK)
            self.to_play = OthelloEnv.WHITE
        return self.state

    def _step(self, action):
        assert self.to_play == self.player_color
        # If already terminal, then don't do anything
        if self.done:
            return self.state, 0., True, {'state': self.state}

        # if HexEnv.pass_move(self.board_size, action):
        #     pass
        
        if OthelloEnv.resign_move(self.board_size, action):
            return self.state, -1, True, {'state': self.state}
        elif not OthelloEnv.valid_move(self.state, self.to_play, action):
            if self.illegal_move_mode == 'raise':
                raise Exception("illegal move exeption")
            elif self.illegal_move_mode == 'lose':
                # Automatic loss on illegal move
                self.done = True
                return self.state, -1., True, {'state': self.state}
            else:
                raise error.Error('Unsupported illegal move action: {}'.format(self.illegal_move_mode))
        else:
            OthelloEnv.make_move(self.state, action, self.player_color)

        # Opponent play
        a = self.opponent_policy(self.state,OthelloEnv.get_opponent(self.player_color))

        # if HexEnv.pass_move(self.board_size, action):
        #     pass

        # Making move if there are moves left
        if a is not None:
            if OthelloEnv.resign_move(self.board_size, action):
                return self.state, 1, True, {'state': self.state}
            else:
                OthelloEnv.make_move(self.state, a, 1 - self.player_color)

        reward = OthelloEnv.game_finished(self.state)
        if self.player_color == OthelloEnv.WHITE:
            reward = - reward
        self.done = reward != 0
        return self.state, reward, self.done, {'state': self.state}

    # def _reset_opponent(self):
    #     if self.opponent == 'random':
    #         self.opponent_policy = random_policy
    #     else:
    #         raise error.Error('Unrecognized opponent policy {}'.format(self.opponent))

    def _render(self, mode='human', close=False):
        if close:
            return
        board = self.state
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        outfile.write(' ' * 5)
        for j in range(board.shape[1]):
            outfile.write(' ' +  str(j + 1) + '  | ')
        outfile.write('\n')
        outfile.write(' ' * 5)
        outfile.write('-' * (board.shape[1] * 6 - 1))
        outfile.write('\n')
        for i in range(board.shape[1]):
            outfile.write(' ' + '  |')
            for j in range(board.shape[1]):
                if board[2, i, j] == 1:
                    outfile.write('  O  ')
                elif board[0, i, j] == 1:
                    outfile.write('  B  ')
                else:
                    outfile.write('  W  ')
                outfile.write('|')
            outfile.write('\n')
            outfile.write(' ' )
            outfile.write('-' * (board.shape[1] * 7 - 1))
            outfile.write('\n')

        if mode != 'human':
            return outfile

    # @staticmethod
    # def pass_move(board_size, action):
    #     return action == board_size ** 2

    @staticmethod
    def resign_move(board_size, action):
        return action == board_size ** 2

#    @staticmethod
#    def valid_move(board, action):
#        coords = OthelloEnv.action_to_coordinate(board, action)
#        if board[2, coords[0], coords[1]] == 1:
#            return True
#        else:
#            return False

    @staticmethod
    def valid_move(game_state,player_color,action):
#        print(OthelloEnv.readable_board(game_state),
#              player_color,
#              OthelloEnv.action_to_coordinate(game_state,action))
        board = game_state 
        board_size = board.shape[2]
        color = player_color
        y,x = OthelloEnv.action_to_coordinate(board, action)
        
        if board[2, y, x] == 0:
            return False
            
        enemy_color = OthelloEnv.get_opponent(color)

        # now check in all directions, including diagonal
        for dy,dx in product(range(-1, 2),range(-1,2)):
            if dy == 0 and dx == 0:
                continue

            # there needs to be >= 1 opponent piece
            # in this given direction, followed by 1 of player's piece
            distance = 1
            yp = (distance * dy) + y
            xp = (distance * dx) + x

            while OthelloEnv.is_in_bounds(xp, yp, board_size) and board[enemy_color,yp,xp] == 1.:
                distance += 1
                yp = (distance * dy) + y
                xp = (distance * dx) + x

            if distance > 1 and OthelloEnv.is_in_bounds(xp, yp, board_size) and board[color,yp,xp] == 1.:
                return True
        return False
            
            
    @staticmethod
    def make_move(board, action, player_color):
        board_size = board.shape[-1]
        y,x = OthelloEnv.action_to_coordinate(board, action)
        board = OthelloEnv.place_stone(board,player_color,[y,x])

        enemy_color = OthelloEnv.get_opponent(player_color)

        to_flip = []
        for dy,dx in product(range(-1, 2),range(-1, 2)):
            if dy == 0 and dx == 0:
                continue

            # there needs to be >= 1 opponent piece
            # in this given direction, followed by 1 of player's piece
            distance = 1
            yp = (distance * dy) + y
            xp = (distance * dx) + x

            flip_candidates = []
            while OthelloEnv.is_in_bounds(xp, yp, board_size) and board[enemy_color,yp,xp] == 1.0:
                flip_candidates.append((xp, yp))
                distance += 1
                yp = (distance * dy) + y
                xp = (distance * dx) + x

            if distance > 1 and OthelloEnv.is_in_bounds(xp, yp, board_size) and board[player_color,yp,xp] == 1.0:
                to_flip.extend(flip_candidates)

        for each in to_flip:
            board = OthelloEnv.flip_stone(board,player_color,each)
        
    @staticmethod
    def place_stone(board, color, coords):
        y,x = coords
        board[color,y,x] = 1.0
        board[2,y,x] = 0.0
        return board

    @staticmethod
    def flip_stone(board, color, coords):
        y,x = coords
        enemy_color = OthelloEnv.get_opponent(color)
        board[color,y,x] =1.0
        board[2,y,x] =0.0
        board[enemy_color,y,x] =0.0
        return board
        
    @staticmethod
    def coordinate_to_action(board, coords):
        return coords[0] * board.shape[-1] + coords[1]

    @staticmethod
    def action_to_coordinate(board, action):
        return action // board.shape[-1], action % board.shape[-1]

    @staticmethod
    def readable_board(board):
        board_size = board.shape[-1]
        readable_board = np.zeros((board_size,board_size))
        for y,x in product(range(board_size),range(board_size)):
            if board[0,y,x] == 1:
                readable_board[y,x] = 8
            if board[1,y,x] == 1:
                readable_board[y,x] = 1
                
        return readable_board

    @staticmethod
    def get_possible_actions(board,color):
        board_size = board.shape[-1]
        actions = []
        for y,x in product(range(board_size),range(board_size) ):
            if OthelloEnv.valid_move(board,color, 
                                     OthelloEnv.coordinate_to_action(board,[y,x])):
                actions.append(OthelloEnv.coordinate_to_action(board,[y,x]))

        return actions
                
    @staticmethod
    def get_opponent(color):
        opponent_colormap = {
            OthelloEnv.BLACK: OthelloEnv.WHITE,
            OthelloEnv.WHITE: OthelloEnv.BLACK,
        }
        return opponent_colormap[color]

    @staticmethod
    def is_in_bounds(x, y, size):
        return 0 <= x < size and 0 <= y < size
  
    @staticmethod
    def get_stone_counts(board):
        black_count, white_count = np.sum(board,axis=(1,2))[:2]
        return black_count, white_count
              
    @staticmethod
    def game_finished(board):
        # Returns 1 if player 1 wins, -1 if player 2 wins and 0 otherwise

        black_count, white_count = OthelloEnv.get_stone_counts(board)
        all_stones = black_count + white_count
        # a full board means no more moves can be made, game over.
        if all_stones == board.shape[-1]**2.:
            if black_count > white_count:
                return 1
            elif black_count == white_count:            
                return 0
            else:
                # tie goes to white
                return -1
    
        # a non-full board can still be game-over if neither player can move.
        black_legal = OthelloEnv.get_possible_actions(board, OthelloEnv.BLACK)
        if black_legal:
            return False
    
        white_legal = OthelloEnv.get_possible_actions(board, OthelloEnv.WHITE)
        if white_legal:
            return False
    
        # neither black nor white has valid moves
        if black_count > white_count:
            return 1
        elif black_count == white_count:            
            return 0
        else:
            # tie goes to white
            return -1