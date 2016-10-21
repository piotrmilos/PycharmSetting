from random import randint

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def size(self):
        return len(self.memory)

    def trim_if_needed(self):
        while len(self.memory) > self.capacity:
            self.memory.pop(0)

    def add(self, el):
        self.memory.append(el)
        self.trim_if_needed()

    def sample_batch(self, batch_size):
        batch = [self.memory[randint(0, len(self.memory) - 1)] for _ in xrange(batch_size)]
        return batch

