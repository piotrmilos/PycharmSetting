import sys

from deepsense import neptune

sys.path.append('deeplearning')
sys.path.append('.')
import ml_lib.matplotlib_hack


from rl.utils import get_eps_greedy_policy, state_to_PIL_img, float_formatter
from rl.optimizers import OPTIMIZERS
from rl.env_utils import AtariRepeatedEnv, atari_postprocess_frame
from collections import OrderedDict
from random import randint, random
from time import sleep
import gym
from bunch import Bunch
import numpy as np
import tensorflow as tf

from ml_lib.base_trainer import SimpleTFlowTrainer
from ml_lib.ml_utils import as_mb, map_get_attr, start_timer, elapsed_time_ms, unpack

from rl.models import get_model_create
from rl.replay_memory import ReplayMemory

DEBUG = False
np.set_printoptions(formatter={'float_kind': float_formatter})

class AgentPolicy(object):
    def select_action(self, state):
        raise NotImplementedError()


class QModelBasedPolicy(object):
    def __init__(self, q_model, eps_greedy):

        self.q_model = q_model
        self.eps_greedy = eps_greedy

    def set_eps_greedy(self, eps_greedy):
        self.eps_greedy = eps_greedy

    def select_action(self, s):
        if random() <= self.eps_greedy:
            # Random
            return self.q_model.sample_action()
        else:
            # Greedy
            q = self.q_model.get_q_on_state(s)
            return np.argmax(q)


class QModel(object):
    def __init__(self, model, actions_n):
        self.model = model
        self.actions_n = actions_n

    def sample_action(self):
        if DEBUG:
            print 'sample_action()'
        return randint(0, self.actions_n - 1)

    def get_q_on_state(self, s):
        res = self.model.eval_fun(s=as_mb(s))['q'][0]
        if DEBUG:
            print res
        return res

    def get_q_on_mb(self, s_mb):
        res = self.model.eval_fun(s=s_mb)['q']
        if DEBUG:
            print res
        return res


    def idx_to_mask(self, n, a):
        z = np.zeros((n,), dtype='float32')
        z[a] = 1
        return z

    def idx_mb_to_mask_mb(self, n, a_mb):
        mb_size = a_mb.shape[0]
        z = np.zeros((mb_size, n), dtype='float32')
        z[np.arange(0, mb_size, 1), a_mb] = 1
        return z

    def update(self, s, a, q_target, glr):
        if DEBUG:
            print 'update', s, a, q_target

        print 'run_train', as_mb(s), as_mb(self.idx_to_mask(self.actions_n, a)), as_mb(q_target)
        return self.model.train_fun(s=as_mb(s),
                                         a_mask=as_mb(self.idx_to_mask(self.actions_n, a)),
                                         q_target=as_mb(q_target),
                                         learning_rate=glr)

    def update_on_mb(self, s_mb, a_mb, q_target_mb, glr):
        if DEBUG:
            print 'update', s_mb, a_mb, q_target_mb

        a_mask_mb = self.idx_mb_to_mask_mb(self.actions_n, a_mb)
        return self.model.train_fun(s=s_mb,
                                         a_mask=a_mask_mb,
                                         q_target=q_target_mb,
                                         learning_rate=glr)



class QLearner(SimpleTFlowTrainer):

    def get_create_netptune_channels_charts_fun(self, args):
        def create_neptune_channels_charts(job):
            FREQ = 10000
            FREQ3 = 500
            channel_names = [
                ('steps_done', FREQ, 'last'),
                ('eps_greedy', FREQ, 'last'),
                ('glr', FREQ, 'last'),
                ('mean_rewards_1', 1),
                ('mean_rewards_2', 1),
                ('episode_length', 30),
                ('learn_time_ms', FREQ),
                ('step_time_ms', FREQ),
                ('step_time_ms_every_eval', self.args.eval_steps),
                ('mean_q_value_replay_mb', FREQ),
                ('mse', FREQ),
                ('mae', FREQ),

                ('mean_non_zero_fc1_frac', FREQ3),
                ('mean_non_zero_conv1_frac', FREQ3),
                ('mean_non_zero_conv2_frac', FREQ3),
                ('mean_fc1', FREQ3),
                ('mean_conv1', FREQ3),
                ('mean_conv2', FREQ3),
            ]

            figures_schema = OrderedDict([
                ('mean_q_value', [
                    ('mean_q_value_replay_mb', 'on replay_mb'),
                ]),

                ('performance', [
                    ('learn_time_ms', 'learn_time_ms'),
                    ('step_time_ms_every_eval', 'step_time_ms_every_eval'),
                    ('step_time_ms', 'step_time_ms'),
                ]),

                ('mean_rewards', [
                    ('mean_rewards_1', 'eps=0.05'),
                    ('mean_rewards_2', 'eps=0.1'),
                ]),
                ('mae', [
                    ('mae', 'mae'),
                ]),
                ('mse', [
                    ('mse', 'mse'),
                ]),
                ('episode_length', [
                    ('episode_length', 'episode_length'),
                ]),
                ('eps_greedy', [
                    ('eps_greedy', 'eps_greedy'),
                ]),
                ('glr', [
                    ('glr', 'glr'),
                ]),


                # "First" version of monitoring functionality
                ('net diagnostics_1', [
                    ('mean_non_zero_fc1_frac', 'mean_non_zero_fc1_frac'),
                    ('mean_non_zero_conv1_frac', 'mean_non_zero_conv1_frac'),
                    ('mean_non_zero_conv2_frac', 'mean_non_zero_conv2_frac'),
                ]),

                ('net diagnostics_2', [
                    ('mean_fc1', 'mean_fc1'),
                    ('mean_conv1', 'mean_conv1'),
                    ('mean_conv2', 'mean_conv2'),
                ]),

            ])
            return self.create_neptune_channels_charts(job, channel_names, figures_schema)
        return create_neptune_channels_charts

    DEBUG = False

    def check_dummy(self, v, v2, dummy):
        if not dummy:
            return v
        else:
            return v2

    def evaluate_policy(self, env, policy, episodes_n=100, visualize=False, save_history=False):
        print 'evaluate_policy'
        res = []
        episodes = []

        for episode_i in xrange(episodes_n):
            history = []
            observation = env.reset()

            timestep = 0
            sum_rewards = 0

            while True:
                if visualize:
                    env.render()
                    sleep(self.args.sleep)
                    # print 'observation', observation
                    # action = env.action_space.sample()
                action = policy.select_action(observation)
                if sum_rewards > 10000:
                    print 'tak, sum_rewards > 100000'
                    break
                    #raise RuntimeError('sum_rewards > 10000')
                prev_observation = observation


                observation, reward, done, info = env.step(action)
                #:print timestep, action
                print timestep, action, reward, done

                if save_history:
                    history.append({'s': prev_observation, 'a': action, 'r': reward})

                # policy.notify_step(action, observation, reward, done, info)

                sum_rewards += reward
                if done:
                    if self.DEBUG:
                        print("Episode finished after {} timesteps".format(timestep + 1))
                    break
                timestep += 1
            episodes.append({'history': history})
            res.append(sum_rewards)

            #if sum_rewards > 10000:
                #raise RuntimeError('sum_rewards > 10000')

        return {'mean_rewards': np.mean(res), 'episodes': episodes}

    # @classmethod
    # def create_parser_from_neptune_cfg(cls, cfg_path):
    #     y = yaml.load(open(cfg_path, 'r'))
    #     parser = argparse.ArgumentParser()
    #     type_map = {
    #         'string': str,
    #         'int': int,
    #         'float': float,
    #         'double': float
    #     }
    #
    #     for param in y['parameters']:
    #         type = param['type']
    #         default = param.get('default', None)
    #         name = param['name']
    #         parser.add_argument('--' + name, type=type_map[type], default=default, help='TODO')
    #     return parser
    #
    # def read_args(self):
    #     def create_parser():
    #         parser = argparse.ArgumentParser()
    #         parser.add_argument('--cfg_path', type=str, default=None, help='TODO')
    #         return parser
    #
    #     dummy_parser = create_parser()
    #     dummy_args = dummy_parser.parse_known_args()[0]
    #     parser = self.create_parser_from_neptune_cfg(dummy_args.cfg_path)
    #     args = parser.parse_known_args()[0]
    #     return args

    def set_up_env(self):
        if self.args.env_name == 'breakout':
            self.env = AtariRepeatedEnv(gym.make('Breakout-v0'), 4, atari_postprocess_frame)
            self.eval_env = AtariRepeatedEnv(gym.make('Breakout-v0'), 4, atari_postprocess_frame)
        elif self.args.env_name == 'lunar_lander':
            self.env = gym.make('LunarLander-v2')
            # NOTE(maciek): there is a bug in lunar lander, can't create two instances
            self.eval_env = self.env
        elif self.args.env_name == 'cartpole':
            self.env = gym.make('CartPole-v0')
            self.eval_env = gym.make('CartPole-v0')
        else:
            raise RuntimeError('Wr env_name!')

    def initialize(self):
        self.set_up_env()

    def do_visualize(self, q_model_1):
        s = self.env.reset()
        behav_policy = QModelBasedPolicy(q_model_1, 0.0)
        timestep = 0
        for step_i in xrange(self.args.steps_n):
            self.env.render()
            print behav_policy.q_model.get_q_on_state(s)
            a = behav_policy.select_action(s)
            s_prim, reward, done, info = self.env.step(a)
            s = s_prim
            if done:
                print("Episode finished after {} timesteps".format(timestep + 1))
                s = self.env.reset()
                timestep = 0
            else:
                timestep += 1
            sleep(self.args.sleep)


    def do_q_learning(self, q_model, q_target_model, copy_ops, eps_greedy_policy):
        behav_policy = QModelBasedPolicy(q_target_model, self.args.eps_greedy_start)

        eval_policy_1 = QModelBasedPolicy(q_model, eps_greedy=0.05)
        eval_policy_2 = QModelBasedPolicy(q_model, eps_greedy=0.1)

        self.replay_memory = ReplayMemory(self.args.replay_memory_size)

        if self.args.do_eval:
            mean_rewards_1 = self.evaluate_policy(self.eval_env, eval_policy_1, episodes_n=self.args.eval_episodes, visualize=self.args.visualize)['mean_rewards']
            print 'mean_rewards', mean_rewards_1

        for var in q_model.model.trainable_variables:
            print var.eval()

        glr = self.args.glr
        timestep, episode_idx = 0, 0
        s = self.env.reset()
        print s.shape

        for step_i in xrange(self.args.steps_n):
            step_timer = start_timer()
            eps_greedy = eps_greedy_policy(step_i)

            self.ts.eps_greedy.add(x=step_i, y=eps_greedy)
            self.ts.glr.add(x=step_i, y=glr)

            behav_policy.set_eps_greedy(eps_greedy)
            if (step_i % self.args.print_debug_steps == 0):
                print 'step_i ', step_i

            if (step_i % self.args.copy_steps == 0) and step_i > 0:
                print self.args.copy_steps
                print 50 * '*** running copy\n'
                print 'step_i ', step_i
                self.sess.run(copy_ops)

            if (step_i % self.args.eval_steps == 0) and step_i > 0:
                print 50 * '*** running eval\n'
                print 'step_i ', step_i
                mean_rewards_1 = self.evaluate_policy(self.eval_env, eval_policy_1, episodes_n=self.args.eval_episodes)['mean_rewards']
                mean_rewards_2 = self.evaluate_policy(self.eval_env, eval_policy_2, episodes_n=self.args.eval_episodes)['mean_rewards']
                print 'mean_rewards_1', mean_rewards_1
                print 'mean_rewards_2', mean_rewards_2

                self.ts.mean_rewards_1.add(x=step_i, y=mean_rewards_1)
                self.ts.mean_rewards_2.add(x=step_i, y=mean_rewards_2)

            if (step_i % self.args.save_checkpoint_steps == 0) and step_i > 0:
                print 'Will save model'
                self.model.save_variables(filename='step_{step}.ckpt'.format(step=step_i))

            if self.args.visualize:
                self.env.render()
                sleep(self.args.sleep)

            a = behav_policy.select_action(s)

            s_prim, reward, done, info = self.env.step(a)

            self.replay_memory.add((Bunch(s=s, s_prim=s_prim, reward=float(reward), done=done, a=a)))


            def learn(mb):
                s_mb = np.stack(map_get_attr('s', mb))
                s_prim_mb = np.stack(map_get_attr('s_prim', mb))
                done_mb = np.stack(map_get_attr('done', mb))
                reward_mb = np.stack(map_get_attr('reward', mb))
                a_mb = np.stack(map_get_attr('a', mb))
                #self.image_channel.send(x=step_i, y=neptune.Image('dupa', description='no', data=state_to_PIL_img(s_mb[0])))

                # NOTE(maciek): only for debug
                #q_predicted_before_update_mb = q_model.get_q_on_mb(s_mb)


                # We will be using q values from target model
                q_value_mb = q_target_model.get_q_on_mb(s_prim_mb)

                # Computing q target
                target_mb = np.max(q_value_mb, axis=1) * self.args.gamma * (1 - done_mb) + reward_mb

                self.ts.mean_q_value_replay_mb.add(x=step_i, y=np.mean(target_mb))

                train_result = q_model.update_on_mb(s_mb, a_mb, target_mb, glr=glr)

                # # NOTE(maciek): only for debug
                # if (step_i % self.args.print_debug_steps == 0):
                #     q_predicted_after_update_mb = q_model.get_q_on_mb(s_mb)
                #
                # if (step_i % self.args.print_debug_steps == 0):
                #     print 'q_predicted_before_update_mb, target_mb, q_predicted_after_update_mb, done'
                #     print_one_line([q_predicted_before_update_mb[np.arange(0, len(mb), 1), a_mb],
                #                     target_mb,
                #                     q_predicted_after_update_mb[np.arange(0, len(mb), 1), a_mb], done_mb])
                return train_result



            if (step_i % self.args.print_debug_steps == 0):
                print 'replay_memory_size', self.replay_memory.size()

            if self.replay_memory.size() > self.args.replay_min_size and (step_i % self.args.learn_steps == 0):
                mb = self.replay_memory.sample_batch(self.args.replay_mb_size)
                learn_timer = start_timer()
                train_result = learn(mb)

                def process_train_result(train_result):
                    self.ts.learn_time_ms.add(x=step_i, y=elapsed_time_ms(learn_timer))

                    mse = train_result['mse']
                    mae = train_result['mae']
                    fc1 = train_result['fc1']
                    #mean_non_zero_fc1 = np.mean(np.sum(fc1 > 0, axis=1))

                    if (step_i % self.args.print_debug_steps == 0):
                        print 'do_learn'
                        print mse, mae, train_result['mean_non_zero_conv1_frac'], train_result['mean_non_zero_conv2_frac'], train_result['mean_non_zero_fc1_frac']
                        print train_result['mean_conv1'], train_result['mean_conv2'], train_result['mean_fc1']

                    self.ts.mean_non_zero_conv1_frac.add(x=step_i, y=train_result['mean_non_zero_conv1_frac'])
                    self.ts.mean_non_zero_conv2_frac.add(x=step_i, y=train_result['mean_non_zero_conv2_frac'])
                    self.ts.mean_non_zero_fc1_frac.add(x=step_i, y=train_result['mean_non_zero_fc1_frac'])

                    self.ts.mean_conv1.add(x=step_i, y=train_result['mean_conv1'])
                    self.ts.mean_conv2.add(x=step_i, y=train_result['mean_conv2'])
                    self.ts.mean_fc1.add(x=step_i, y=train_result['mean_fc1'])

                    #print train_result['q_s_a']

                    self.ts.mse.add(x=step_i, y=mse)
                    self.ts.mae.add(x=step_i, y=mae)

                process_train_result(train_result)


            s = s_prim

            if done:
                # if self.DEBUG:
                print("Episode {episode_idx} finished after {steps} timesteps".format(episode_idx=episode_idx,
                                                                                      steps=timestep + 1))
                self.ts.episode_length.add(timestep + 1)
                s = self.env.reset()
                timestep = 0
                episode_idx += 1
            else:
                timestep += 1

            self.ts.step_time_ms_every_eval.add(x=step_i, y=elapsed_time_ms(step_timer))
            self.ts.step_time_ms.add(x=step_i, y=elapsed_time_ms(step_timer))
            self.ts.steps_done.add(step_i + 1)

    def go(self, sess=None):
        self.initialize()
        self.sess = sess
        ACTIONS_N = self.args.actions_n

        model_create_conf_1 = {'actions_n': ACTIONS_N, 'name': 'model_1', 'optimizer_creator': OPTIMIZERS[self.args.optimizer]}

        # We ONLY use the second model as target model, no actually optimizer will be run
        model_create_conf_2 = {'actions_n': ACTIONS_N, 'name': 'model_2', 'optimizer_creator': OPTIMIZERS[self.args.optimizer]}

        tf.set_random_seed(self.args.seed)

        self.model = get_model_create(self.args.model_name)(model_create_conf_1).initialize()
        self.target_model = get_model_create(self.args.model_name)(model_create_conf_2).initialize()
        self.model.create_variables_saver(self.exp_dir_path)
        self.target_model.create_variables_saver(self.exp_dir_path)

        if self.args.restore_checkpoint_path:
            self.model.restore_variables(self.args.restore_checkpoint_path)

        def create_copy_model1_to_model2_ops():
            vars1 = sorted(self.model.trainable_variables, key=lambda a: a.name)
            vars2 = sorted(self.target_model.trainable_variables, key=lambda a: a.name)
            assert(len(vars1) == len(vars2))
            ops = []
            for var1, var2 in zip(vars1, vars2):
                ops.append(var2.assign(var1))
            return ops

        q_model = QModel(self.model, ACTIONS_N)
        q_target_model = QModel(self.target_model, ACTIONS_N)
        copy_ops = create_copy_model1_to_model2_ops()
        self.sess.run(copy_ops)

        eps_greedy_policy = get_eps_greedy_policy(self.args.eps_greedy_start, self.args.eps_greedy_end, self.args.eps_greedy_steps_n)

        print 'go()'

        if self.args.do_visualize:
            self.do_visualize(q_model)

        if self.args.do_train:
            self.do_q_learning(q_model, q_target_model, copy_ops, eps_greedy_policy)


if __name__ == '__main__':
    learner = QLearner()
    learner._main()
