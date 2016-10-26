#!/usr/bin/env python3
import time
import matplotlib.pyplot as plt

plt.ion()
mng = plt.get_current_fig_manager()

while True:
    raw = open('neural/results_train.txt', 'r')
    lines = raw.readlines()

    as_float = [float(line) for line in lines]

    plt.clf()
    plt.ylim((0,100))
    plt.plot(as_float)

    plt.yticks(range(0, 100, 10))

    plt.draw()
    raw.close()

    for i in range(10, 0, -1):
        print(i)
        plt.pause(1)

