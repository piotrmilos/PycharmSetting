import numpy as np
import random
from util import to_offset, numpify, best_move_val
import math

# MAX_MEM_LEN = 1000  # no matter what, do not allow memory past this amount


class ExperienceReplay:

    def __init__(self, size):
        self.memory = []
        self.MEM_LEN = size

    def remember(self, S, a, r, Sp, l, win):
        self.memory.append((S, a, r, Sp, l, win))
        if len(self.memory) > self.MEM_LEN:
            self.memory.pop(0)

    def set_replay_len(self, val):
        if val == -1:
            self.MEM_LEN = float('inf')
        elif val < 0:
            print('invalid memory length: {}'.format(val))
            quit()
        else:
            self.MEM_LEN = val

    def get_replay(self, model, batch_size, ALPHA):
        # (S, a, r, Sp, legal, win) tuple
        S, a, r, Sp, l, win = range(6)  # indices

        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        replays = random.sample(self.memory, batch_size)

        # now format for training
        board_size = model.input_shape[1]
        inputs = np.empty((batch_size, board_size))
        targets = np.empty((batch_size, board_size))
        for index, replay in enumerate(replays):
            if replay[win] is False and not replay[l]:
                continue  # no legal moves, and not a win
            move = int(to_offset(replay[a], math.sqrt(board_size)))
            state = numpify(replay[S])
            state_prime = numpify(replay[Sp])
            prev_qvals = model.predict(state)

            q_prime = None
            if replay[win] is False:
                next_qvals = model.predict(state_prime)
                _, best_q = best_move_val(next_qvals, replay[l])
                # q_prime = (1 - ALPHA) * \
                #  prev_qvals[0][move] + ALPHA * (replay[r] + best_q)
                q_prime = replay[r] + best_q
            else:
                # q_prime = (1 - ALPHA) * prev_qvals[0][move] + ALPHA * replay[r]
                q_prime = replay[r]
            prev_qvals[0][move] = q_prime
            inputs[index] = state
            targets[index] = prev_qvals

        return inputs, targets

    def __len__(self):
        return len(self.memory)
