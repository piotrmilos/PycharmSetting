class TimeSeries(object):
    def __getstate__(self):
        return {'t': self.t}

    def __savestate__(self, d):
        self.t = d['t']

    def __init__(self):
        self.t = []
        self.t_x = []

        self.add_observers = []

    def add(self, y, x=None):
        if x is None:
            x = (0 if len(self.t_x) == 0 else self.t_x[-1]) + 1

        self.t.append(y)
        self.t_x.append(x)

        for add_observer in self.add_observers:
            add_observer.notify_add(self, y, x)

    def add_add_observer(self, observer):
        self.add_observers.append(observer)

    def size(self):
        return len(self.t)

    def last_mean(self, n=None):
        if n is None:
            n = self.size()

        if n > self.size():
            raise RuntimeError()

        return np.mean(self.t[-n:])

    def get_items(self):
        return self.t

    def last_n(self, n):
        if n == -1:
            return self.t
        if n > self.size():
            raise RuntimeError()

        return self.t[-n:]

    def last_x(self):
        return self.t_x[-1]


import numpy as np


class NeptuneTimeseriesObserver(object):
    @classmethod
    def last(cls, l):
        return l[-1]

    @classmethod
    def mean(cls, l):
        return np.mean(l)

    def __init__(self, name, channel, add_freq, fun=None):
        if fun is None:
            fun = self.mean

        self.name = name
        self.add_freq = add_freq
        self.fun = fun
        self.channel = channel

    def notify_add(self, ts, y, x):
        if ts.size() % self.add_freq == 0 and ts.size():
            #print 'Channel store_objects!'
            #print 'name ', self.name
            X = float(x)
            Y = float(self.fun(ts.last_n(self.add_freq)))
            #print X, Y
            #print type(X), type(Y)
            self.channel.send(x=X, y=Y)
