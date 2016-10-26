import random
from agent import Agent
from util import *

class RandomAgent(Agent):
    """An agent that simply chooses
    totally random legal moves."""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color

    def get_action(self, state, legal_moves):
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def reset(self):
        pass

    def observe_win(self, winner):
        pass
