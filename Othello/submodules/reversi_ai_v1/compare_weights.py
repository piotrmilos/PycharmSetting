#!/usr/bin/env python3
from sys import argv
import run_game
from prop_parse import prop_parse

AMOUNT = 1336

output_file = 'results.txt'


def main():

    input_args = prop_parse(argv)
    input_args['weights_file'] = 'neural/q_weights'

    f = open(output_file, 'w')

    summary = []
    for i in range(1, AMOUNT + 1):
        input_args['weights_num'] = '_' + str(float(i))
        result = run_game.main(**input_args)
        message = str(result['Black'])
        print('weight {} black won {:.2f}'.format(i, result['Black']))
        summary.append(result)
        f.write(message + '\n')

    for index, result in enumerate(summary):
        print('{:.2f}% - weight {}'.format(result['Black'], index))

    f.close()


if __name__ == '__main__':
    main()
