
import argparse

import gym
import sys
import numpy as np

from TensorLib.helpers import create_variable

DEBUG = False


class LinearCartpolePolicy(object):
    def __init__(self, params):
        self.params = params

    def set_params(self, params):
        self.params = params

    def notify_reset(self, observation):
        self.last_observation = observation

    def notify_step(self, action_performed, observation, reward, done, info):
        self.last_observation = observation

    def select_action(self):
        z = float(np.matmul(self.last_observation, self.params))
        if z > 0:
            action = 0
        else:
            action = 1
        return action

import tensorflow as tf

class LinearCartpolePolicy2(object):
    def create_model(self):
        W = 0.1
        observation = tf.placeholder(tf.float32, (4,), name='observation')
        params = tf.get_variable('params', initializer=tf.random_uniform((4,2), minval=-W, maxval=W))
        print observation.get_shape(), params.get_shape()
        probs = tf.nn.softmax(tf.matmul(tf.reshape(observation, (1, 4)),
                                        tf.reshape(params, (4, 2))))
        probs = tf.squeeze(probs)
        #probs = tf.pack([probs, 1 - probs])

        action = tf.placeholder(tf.int32, ())

        print probs.get_shape()
        print action.get_shape()
        r = tf.slice(probs, tf.reshape(action, (1,)), [1,])
       # r = probs[action]
        grad = tf.gradients(ys=r, xs=[params])[0]

        def get_probs(observation_):
            return probs.eval(feed_dict={observation: observation_})

        def get_grad(observation_, action_):
            return grad.eval(feed_dict={observation: observation_, action: action_})


        new_params = tf.placeholder(tf.float32)
        assign_params = params.assign(new_params)

        def set_params(new_params_):
            return assign_params.eval(feed_dict={new_params: new_params_})

        def get_params():
            return params.eval()



        return {
            'params': params,
            'get_probs': get_probs,
            'get_grad': get_grad,
            'set_params': set_params,
            'get_params': get_params
        }


    def init_model_params(self):
        tf.initialize_all_variables().run()

    def __init__(self):
        self.model = self.create_model()
        self.get_probs = self.model['get_probs']
        self.get_grad = self.model['get_grad']
        self.set_params = self.model['set_params']
        self.get_params = self.model['get_params']


    def set_params(self, params):
        raise NotImplementedError()
        self.params = params

    def notify_reset(self, observation):
        self.last_observation = observation

    def notify_step(self, action_performed, observation, reward, done, info):
        self.last_observation = observation

    def get_action_probs(self, observation):
        return self.get_probs(observation)

    def select_action(self):
        def sample_action(probs):
            a = np.random.random()
            for idx in xrange(len(probs)):
                if a <= probs[idx]:
                    return idx
                a -= probs[idx]
            return 0

        probs = self.get_action_probs(self.last_observation)
        return sample_action(probs)


    def grad(self, observation, action):
        return self.get_grad(observation, action)

        #print 'grad'
        #print observation.shape, self.params.shape
        u = np.matmul(observation, self.params)
        d_res_d_z = 1 if action == 0 else -1
        d_res_d_u = d_res_d_z * d_sigmoid(u)
        #print d_res_d_u
        #print d_res_d_u.shape
        #print observation
        d_res_d_params = d_res_d_u * observation
        return d_res_d_params

    def grad_finite(self, observation, action):
        #print 'grad_finite'
        old_params = np.copy(self.params)
        diff = 0.01
        res = self.get_action_probs(observation)
        grad = np.zeros_like(old_params)

        for idx in xrange(len(self.params)):
            self.params[idx] = old_params[idx] + diff
            res2 = self.get_action_probs(observation)
            grad[idx] = (res2[action] - res[action]) / diff
            self.params[idx] = old_params[idx]
        return grad


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class CartpoleReinforceTrainer(object):
    def evaluate_policy(self, env, policy, episodes_n=100, visualize=False, save_history=False):
        res = []
        episodes = []

        for episode_i in xrange(episodes_n):
            history = []
            observation = env.reset()
            policy.notify_reset(observation)

            timestep = 0
            sum_rewards = 0

            while True:
                if visualize:
                    env.render()
                #print 'observation', observation
               #action = env.action_space.sample()
                action = policy.select_action()
                if sum_rewards > 10000:
                    print 'tak'
                    break
                prev_observation = observation
                observation, reward, done, info = env.step(action)
                if save_history:
                    history.append({'observation': prev_observation, 'action': action, 'reward': reward})

                policy.notify_step(action, observation, reward, done, info)

                sum_rewards += reward
                if done:
                    if DEBUG:
                        print("Episode finished after {} timesteps".format(timestep + 1))
                    break
                timestep += 1
            episodes.append({'history': history})
            res.append(sum_rewards)
            if sum_rewards > 10000:
                break

        return {'mean_rewards': np.mean(res), 'episodes': episodes}

    def create_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=None, help='TODO')
        parser.add_argument('--policy_grad', default=None, help='TODO', action='store_true')
        parser.add_argument('--policy_sampling', default=None, help='TODO', action='store_true')
        return parser


    def go(self):
        parser = self.create_parser()
        self.args = parser.parse_args()

        with tf.Session() as sess:
            if self.args.policy_grad:
                self.find_policy_by_policy_gradient()

            if self.args.policy_sampling:
                self.find_policy_by_sampling()

    def find_policy_by_policy_gradient(self):


        env = gym.make('CartPole-v0')
        policy = LinearCartpolePolicy2()
        policy.init_model_params()

        res = self.evaluate_policy(env, policy, 1000, visualize=False, save_history=True)
        print res['mean_rewards']


        # observation = env.reset()
        # policy.notify_reset(observation)
        # print 'observation', observation
        # grads1 = [policy.grad(observation, 0), policy.grad(observation, 1)]
        # grads2 = [policy.grad_finite(observation, 0), policy.grad_finite(observation, 1)]
        # print grads1
        # print grads2


        learning_rate = self.args.lr
        iter_n = 100

        for iter_i in xrange(iter_n):
            episodes_n = 200

            res = self.evaluate_policy(env, policy, episodes_n, visualize=False, save_history=True)
            new_params = policy.get_params()
            print new_params

            sum_rewards = []
            print res['mean_rewards']
            for episode_i in xrange(episodes_n):
                history = res['episodes'][episode_i]['history']

                sum_rew = 0


                #for a in history:
                for a in reversed(history):
                    observation = a['observation']
                    action = a['action']

                    sum_rew += a['reward']
                    Q_s_a_sample = sum_rew

                    p_a_s = policy.get_action_probs(observation)[action]
                    #if episode_i % 200 == 0:
                        #print policy.get_action_probs(observation)
                    g = policy.grad(observation, action)
                    #g = policy.grad_finite(observation, action)
                    p_a_s = np.clip(p_a_s, 0.0000001, 1.0)
                    grad_log_a_s = (1.0 / p_a_s) * g

                    #if episode_i % 200 == 0:
                        #print grad_log_a_s

                    grad_log_a_s = np.clip(grad_log_a_s, -1.0, 1.0)
                    #if episode_i % 200 == 0:
                        #print grad_log_a_s


                    # WTF? why - works, not +
                    update = learning_rate * Q_s_a_sample * grad_log_a_s
            #        print update
                    new_params += update
                    #params = params + learning_rate * Q_s_a_sample * grad_log_a_s
                sum_rewards.append(sum_rew)

            policy.set_params(new_params)

            if True:
                print 'sum_rew', np.mean(sum_rewards)
                print new_params

        #        print 'eval_policy', self.evaluate_policy(env, policy, 4, visualize=True)['mean_rewards']
            # if episode_i == 3000:,
            #     learning_rate *= 10
            #     print 'decrease lr'



    def find_policy_by_sampling(self):
        param_samples_n = 10000
        env = gym.make('CartPole-v0')

        res = []
        best_mean_reward = 0
        for sample_idx in xrange(param_samples_n):
            W = 100
            params = np.random.uniform(-W, W, (4,))
            policy = LinearCartpolePolicy2(params)
            mean_reward = self.evaluate_policy(env, policy)['mean_rewards']
            print sample_idx, params, mean_reward, best_mean_reward
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
            res.append((policy, mean_reward))

            if mean_reward > 1000:
                print 'Found params with mean_reward =', mean_reward
                print 'breaking'
                break
        res = sorted(res, key=lambda a: -a[1])
        policy = res[0][0]
        print policy.params

        print self.evaluate_policy(env, policy, 100, visualize=False)
        #print self.evaluate_policy(env, res[0][0], 1000)


def main():
    trainer = CartpoleReinforceTrainer()
    sys.exit(trainer.go())


if __name__ == '__main__':
    main()


