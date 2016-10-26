from agents.agent import Agent
from agents import MonteCarloAgent

class HumanAgent(Agent):
    """This agent is controlled by a human, who inputs moves via stdin."""

    def __init__(self, reversi, color, **kwargs):
        self.reversi = reversi
        self.color = color

    def reset(self):
        pass

    def observe_win(self, winner):
        pass

    def get_action(self, game_state, legal):
        if not legal:
            return None
        choice = None
        while True:
            raw_choice = input('Enter a move x,y: ')
            if raw_choice == 'pass':
                return None
            elif raw_choice == 'exit' or raw_choice == 'quit':
                quit()
            elif raw_choice.startswith('helpme'):
                sim_time = 5
                s = raw_choice.split()
                if len(s) == 2 and s[1].isdigit():
                    sim_time = int(s[1])
                self.get_help(game_state, legal, sim_time)
                continue
            elif len(raw_choice) != 3:
                print('input must be 3 long, formatted x,y')
                continue


            if raw_choice[1] != ',':
                print('comma separator not found.')
                continue
            if not raw_choice[0].isdigit() or not raw_choice[2].isdigit():
                print('couldn\'t determine x,y from your input.')
                continue
            choice = (int(raw_choice[0]), int(raw_choice[2]))
            if choice not in legal:
                print('not a legal move. try again.')
                continue
            else:
                break

        return choice

    def get_help(self, game_state, legal, sim_time):
        mc = MonteCarloAgent(self.reversi, self.color, sim_time=sim_time)
        action = mc.get_action(game_state, legal)
        print('suggested move: {}'.format(action))
