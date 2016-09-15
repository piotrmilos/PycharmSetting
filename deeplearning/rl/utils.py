from ml_lib.image_utils import numpy_img_to_PIL


def get_eps_greedy_policy(eps_greedy_start, eps_greedy_end, eps_greedy_steps_n):
    def eps_greedy_policy(step_i):
        if step_i < eps_greedy_steps_n:
            eps_greedy = ((eps_greedy_start * (1 - step_i / float(eps_greedy_steps_n))) +
                              (eps_greedy_end * (step_i / float(eps_greedy_steps_n))))
        else:
            eps_greedy = eps_greedy_end
        return eps_greedy
    return eps_greedy_policy

import numpy as np
def state_to_PIL_img(state):
    S = 84
    assert(state.shape == (4, S, S))
    img = np.zeros((84 * 2, 84 * 2), dtype='uint8')
    img[0 * S: 1 * S, 0 * S: 1 * S] = state[0, ...]
    img[1 * S: 2 * S, 0 * S: 1 * S] = state[1, ...]
    img[0 * S: 1 * S, 1 * S: 2 * S] = state[2, ...]
    img[1 * S: 2 * S, 1 * S: 2 * S] = state[3, ...]
    return numpy_img_to_PIL(img)

float_formatter = lambda x: "%.4f" % x

def print_one_line(arrs):
        n = arrs[0].shape[0]
        for i in xrange(n):
            z = []
            for arr in arrs:
                r = arr[i]
                z.append(float_formatter(r))
            print ' '.join(z)


