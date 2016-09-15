import numpy as np

from PIL import Image

def atari_postprocess_frame(arr):
    img = Image.fromarray(arr)
    img = img.resize((84, 84)).convert('L')
    processed_observation = np.array(img)
    return processed_observation.astype('uint8')

class AtariRepeatedEnv(object):
    def __init__(self, orig_env, k, frame_posprocess_fun=None):
        self.orig_env = orig_env
        self.k = k
        if frame_posprocess_fun is not None:
            self.frame_postprocess_fun = frame_posprocess_fun
        else:
            self.frame_postprocess_fun = lambda a: a


    def step(self, action):
        # WARNING(maciek): Maybe we should be more precise here at the terminations of the episode,
        # not sure what happens now, i.e episode ended and we still make steps!
        observations, rewards, dones, infos = [], [], [], []
        for i in xrange(self.k):
            observation, reward, done, info = self.orig_env.step(action)
            observation = self.frame_postprocess_fun(observation)

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        return np.stack(observations), sum(rewards), (sum(dones) > 1), infos

    def render(self):
        return self.orig_env.render()

    def reset(self):
        return np.stack(self.k * [self.frame_postprocess_fun(self.orig_env.reset())])



