"""This Q-Learning neural network agent is still a work in progress and is not complete yet."""
import random
from agents import Agent
from keras.layers import Dense
from keras.models import Sequential, model_from_json
from keras.optimizers import RMSprop, SGD
from util import info, opponent, color_name, numpify, best_move_val

MODEL_FILENAME = 'neural/q_model'
WEIGHTS_FILENAME = 'neural/q_weights'
HIDDEN_SIZE = 42
ALPHA = 1.0
BATCH_SIZE = 64

WIN_REWARD = 1
LOSE_REWARD = -1
optimizer = RMSprop()
# optimizer = SGD(lr=0.01, momentum=0.0)


class QLearningAgent(Agent):

    def __init__(self, reversi, color, **kwargs):
        self.color = color
        self.reversi = reversi
        self.learning_enabled = kwargs.get('learning_enabled', False)
        self.model = self.get_model(kwargs.get('model_file', None))
        self.minimax_enabled = kwargs.get('minimax', False)

        weights_num = kwargs.get('weights_num', '')
        self.load_weights(weights_num)

        # training values
        self.epsilon = 0.0
        if self.learning_enabled:
            self.epoch = 0
            self.train_count = random.choice(range(BATCH_SIZE))
            self.memory = None
            self.prev_move = None
            self.prev_state = None

        if kwargs.get('model_file', None) is None:
            # if user didn't specify a model file, save the one we generated
            self.save_model(self.model)

    def set_epsilon(self, val):
        self.epsilon = val
        if not self.learning_enabled:
            info('Warning -- set_epsilon() was called when learning was not enabled.')

    def set_memory(self, memory):
        self.memory = memory

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_action(self, state, legal_moves=None):
        """Agent method, called by the game to pick a move."""
        if legal_moves is None:
            legal_moves = self.reversi.legal_moves(state)

        if not legal_moves:
            # no actions available
            return None
        else:
            move = None
            if self.epsilon > random.random():
                move = random.choice(legal_moves)
            else:
                move = self.policy(state, legal_moves)
            if self.learning_enabled:
                self.train(state, legal_moves)
                self.prev_move = move
                self.prev_state = state
            return move

    def minimax(self, state, depth=2, alpha=-float('inf'), beta=float('inf')):
        # pdb.set_trace()
        """Given a state, find its minimax value."""
        assert state[1] == self.color or state[1] == opponent[self.color]
        player_turn = True if state[1] == self.color else False

        legal = self.reversi.legal_moves(state)
        winner = self.reversi.winner(state)
        if not legal and winner is False:
            # no legal moves, but game isn't over, so pass turn
            return self.minimax(self.reversi.next_state(state, None))
        elif depth == 0 or winner is not False:
            if winner == self.color:
                return 9999999
            elif winner == opponent[self.color]:
                return -9999999
            else:
                q_vals = self.model.predict(numpify(state))
                best_move, best_q = best_move_val(q_vals, legal)
                print('best_q: {}'.format(best_q))
                return best_q

        if player_turn:
            val = -float('inf')
            for move in legal:
                new_state = self.reversi.next_state(state, move)
                val = max(val, self.minimax(new_state, depth - 1, alpha, beta))
                alpha = max(alpha, val)
                if beta <= alpha:
                    break
            return val
        else:
            val = float('inf')
            for move in legal:
                new_state = self.reversi.next_state(state, move)
                val = min(val, self.minimax(new_state, depth - 1, alpha, beta))
                beta = min(beta, val)
                if beta <= alpha:
                    break
            return val

    def observe_win(self, state):
        """Called by the game at end of game to present the agent with the final board state."""
        if self.learning_enabled:
            winner = self.reversi.winner(state)
            self.train(state, [], winner)

    def reset(self):
        """Resets the agent to prepare it to play another game."""
        self.reset_learning()

    def reset_learning(self):
        self.prev_move = None
        self.prev_state = None

    def policy(self, state, legal_moves):
        """The policy of picking an action based on their weights."""
        if not legal_moves:
            return None

        if not self.minimax_enabled:
            # don't use minimax if we're in learning mode
            best_move, _ = best_move_val(
                self.model.predict(numpify(state)),
                legal_moves
            )
            return best_move
        else:
            next_states = {self.reversi.next_state(
                state, move): move for move in legal_moves}
            move_scores = []
            for s in next_states.keys():
                score = self.minimax(s)
                move_scores.append((score, s))
                info('{}: {}'.format(next_states[s], score))

            best_val = -float('inf')
            best_move = None
            for each in move_scores:
                if each[0] > best_val:
                    best_val = each[0]
                    best_move = next_states[each[1]]

            assert best_move is not None
            return best_move

    def train(self, state, legal_moves, winner=False):
        assert self.memory is not None, "can't train without setting memory first"
        self.train_count += 1
        model = self.model
        if self.prev_state is None:
            # on first move, no training to do yet
            self.prev_state = state
        else:
            # add new info to replay memory
            reward = 0
            if winner == self.color:
                reward = WIN_REWARD
            elif winner == opponent[self.color]:
                reward = LOSE_REWARD
            elif winner is not False:
                raise ValueError

            self.memory.remember(self.prev_state, self.prev_move,
                                 reward, state, legal_moves, winner)

            # get an experience from memory and train on it
            if self.train_count % BATCH_SIZE == 0 or winner is not False:
                states, targets = self.memory.get_replay(
                    model, BATCH_SIZE, ALPHA)
                model.train_on_batch(states, targets)

    def get_model(self, filename=None):
        """Given a filename, load that model file; otherwise, generate a new model."""
        model = None
        if filename:
            info('attempting to load model {}'.format(filename))
            try:
                model = model_from_json(open(filename).read())
            except FileNotFoundError:
                print('could not load file {}'.format(filename))
                quit()
            print('loaded model file {}'.format(filename))
        else:
            print('no model file loaded, generating new model.')
            size = self.reversi.size ** 2
            model = Sequential()
            model.add(Dense(HIDDEN_SIZE, activation='relu', input_dim=size))
            # model.add(Dense(HIDDEN_SIZE, activation='relu'))
            model.add(Dense(size))

        model.compile(loss='mse', optimizer=optimizer)
        return model

    @staticmethod
    def save_model(model):
        """Given a model, save it to disk."""
        as_json = model.to_json()
        with open(MODEL_FILENAME, 'w') as f:
            f.write(as_json)
            print('model saved to {}'.format(MODEL_FILENAME))

    def save_weights(self, suffix):
        filename = '{}{}{}{}'.format(WEIGHTS_FILENAME, color_name[
            self.color], suffix, '.h5')
        print('saving weights to {}'.format(filename))
        self.model.save_weights(filename, overwrite=True)

    def load_weights(self, suffix):
        filename = '{}{}{}{}'.format(WEIGHTS_FILENAME, color_name[
            self.color], suffix, '.h5')
        print('loading weights from {}'.format(filename))
        try:
            self.model.load_weights(filename)
        except:
            print('Couldn\'t load weights file {}!'.format(filename))
