#!/usr/bin/env python3
import time
from agents.q_learning_agent import QLearningAgent
from agents import RandomAgent
from game.reversi import Reversi
from math import floor
from util import *
from agents import ExperienceReplay

import sys

SNAPSHOT_AMNT = 2000  # this frequently, save a snapshot of the states
STOP_EXPLORING = 0.3  # after how many games do we set epsilon to 0?
TEST_GAMES = 1000
DATA_FILE = 'neural/results_train.txt'

BOARD_SIZE = 8
REPLAY_MEM = 2000
MIN_EPSILON = 0.01


def main():
    reset_output_file()
    amount = 40000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        amount = int(sys.argv[1])

    reversi = Reversi(size=BOARD_SIZE, WhiteAgent=QLearningAgent,
            BlackAgent=QLearningAgent, silent=True, learning_enabled=True)

    black_mem = ExperienceReplay(REPLAY_MEM)
    white_mem = ExperienceReplay(REPLAY_MEM)
    reversi.black_agent.set_memory(black_mem)
    reversi.white_agent.set_memory(white_mem)

    epsilon = 1.0
    end_exploration = max(1, floor(amount * STOP_EXPLORING))
    print('exploration will halt at {} games.'.format(end_exploration))

    start = time.time()
    try:
        for i in range(1, amount + 1):
            print('playing game {}/{} ({:3.2f}%) epsilon: {:.2f}'.format(i,
                amount, i * 100 / amount, epsilon))
            reversi.white_agent.set_epsilon(epsilon)
            reversi.black_agent.set_epsilon(epsilon)
            reversi.black_agent.memory.set_replay_len(min(i, REPLAY_MEM))
            reversi.white_agent.memory.set_replay_len(min(i, REPLAY_MEM))
            reversi.white_agent.set_epoch(i)
            reversi.black_agent.set_epoch(i)
            reversi.play_game()

            if i % SNAPSHOT_AMNT == 0:
                amnt = i / SNAPSHOT_AMNT
                reversi.white_agent.save_weights('_' + str(amnt))
                reversi.black_agent.save_weights('_' + str(amnt))
                play_test_games(amnt)

            epsilon -= (1 / end_exploration)
            epsilon = max(epsilon, MIN_EPSILON)
    except KeyboardInterrupt:
        print('Stopping.  Will save weights before quitting.')

    seconds = time.time() - start
    print('time: {:.2f} minutes. per game: {:.2f}ms.'.format(
        seconds / 60.0, (seconds / float(i)) * 1000.0))
    reversi.white_agent.save_weights('')
    reversi.black_agent.save_weights('')

def reset_output_file():
    with open(DATA_FILE, 'w') as f:
        f.write('')


def play_test_games(weight_num):
    print('playing test games...')
    wincount = 0
    testgame = Reversi(size=BOARD_SIZE, WhiteAgent=RandomAgent, BlackAgent=QLearningAgent, minimax=False, silent=True,
            model_file='neural/q_model', model_weights='neural/q_weights', weights_num='_' + str(weight_num))
    for i in range(TEST_GAMES):
        print('playing test game {}/{}'.format(i, TEST_GAMES))
        winner, _, _ = testgame.play_game()
        print('winner: {} black: {}'.format(winner, BLACK))
        if winner == BLACK:
            wincount += 1

    winrate = 100.0 * wincount / TEST_GAMES
    result = '{}: {:.2f}%\n'.format(weight_num, winrate)
    print('result: {}'.format(result))
    with open(DATA_FILE, 'a') as f:
        f.write(str(winrate) + '\n')
    print('done with test games.')


if __name__ == "__main__":
    main()
