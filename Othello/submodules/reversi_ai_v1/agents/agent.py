class Agent:
    """An abstract class defining the interface for a Reversi agent."""

    def __init__(self, reversi, color):
        raise NotImplementedError

    def get_action(self, game_state, legal_moves=None):
        raise NotImplementedError

    def observe_win(self, state, winner):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
