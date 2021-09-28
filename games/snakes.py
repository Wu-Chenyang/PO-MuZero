import datetime
import os

import gym
import numpy as np
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # More information is available here: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization

        self.seed = 0  # Seed for numpy, torch and the game
        self.max_num_gpus = None  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available



        ### Game
        self.observation_shape = (5, 12, 22)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
        self.action_space = list(range(64))  # Fixed list of all possible actions. You should only edit the length
        self.players = list(range(2))  # List of players. You should only edit the length
        self.stacked_observations = 0  # Number of previous observations and previous actions to add to the current observation

        # If the game is partially observable, the environment will return an observation for each player at each step. 
        # All observations in a round will be concatenated to be a round-observation. The round observation is of the shape
        # `observation_shape`. If a `None` observation is returned, it will be omitted when generating round-observations.
        # The round-observation and the action taken by the agent in this round will be concatenated to form an a-o pair.
        # `stack_observations` specify the number of a-o pair concatenated to form the history.
        self.partially_observable = True

        # Evaluate
        self.muzero_player = 1  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
        self.opponent = "random"  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class

        ### Self-Play
        self.num_workers = 12  # Number of simultaneous threads/workers self-playing to feed the replay buffer
        self.selfplay_on_gpu = False
        self.max_moves = 400  # Maximum number of moves if game is not finished before
        self.num_simulations = 100  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.temperature_threshold = None  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.network = "resnet"  # "resnet" / "fullyconnected"
        self.support_size = 20  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        
        # Residual Network
        self.downsample = False  # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
        self.blocks = 3  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        self.reduced_channels_reward = 2  # Number of channels in reward head
        self.reduced_channels_value = 2  # Number of channels in value head
        self.reduced_channels_policy = 4  # Number of channels in policy head
        self.resnet_fc_reward_layers = [64]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [64]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [64]  # Define the hidden layers in the policy head of the prediction network

        # Fully Connected Network
        self.encoding_size = 32
        self.fc_representation_layers = [64]  # Define the hidden layers in the representation network
        self.fc_dynamics_layers = [64]  # Define the hidden layers in the dynamics network
        self.fc_reward_layers = [64]  # Define the hidden layers in the reward network
        self.fc_value_layers = [64]  # Define the hidden layers in the value network
        self.fc_policy_layers = [64]  # Define the hidden layers in the policy network


        ### Training
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../results", os.path.basename(__file__)[:-3], datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights and TensorBoard logs
        self.save_model = True  # Save the checkpoint in results_path as model.checkpoint
        self.training_steps = 300000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = 64  # Number of parts of games to train on at each training step
        self.checkpoint_interval = 50  # Number of training steps before using the model for self-playing
        self.value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.train_on_gpu = torch.cuda.is_available()  # Train on GPU if available

        self.optimizer = "Adam"  # "Adam" or "SGD". Paper uses SGD
        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9  # Used only if optimizer is SGD

        # Exponential learning rate schedule
        self.lr_init = 0.002  # Initial learning rate
        self.lr_decay_rate = 0.9  # Set it to 1 to use a constant learning rate
        self.lr_decay_steps = 10000


        ### Replay Buffer
        self.replay_buffer_size = 10000  # Number of self-play games to keep in the replay buffer
        self.num_unroll_steps = 50  # Number of game moves to keep for every batch element
        self.td_steps = 30  # Number of steps in the future to take into account for calculating the target value
        self.PER = True  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
        self.PER_alpha = 0.5  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

        # Reanalyze (See paper appendix Reanalyse)
        self.use_last_model_value = False  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.reanalyse_on_gpu = False



        ### Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        self.ratio = 1.5  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to None to disable it


    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

import json
import random
from . import _snakes
import os

def get_vec_action(action):
    a1 = action % 4
    action = (action - a1) // 4
    a2 = action % 4
    a3 = (action - a2) // 4
    return np.array([a1, a2, a3])

def get_observation(state):
    snakes_positions = {key-2: state[key] for key in {2, 3, 4, 5, 6, 7}}

    o_index_min = 3 if state['controlled_snake_index'] > 4 else 0
    controlled_snake_indices = [o_index_min, o_index_min+1, o_index_min+2]

    obs = np.zeros((5, state['board_height']+2, state['board_width']+2))

    for index, pos in snakes_positions.items():
        if index in controlled_snake_indices:
            for i, p in enumerate(pos):
                obs[index if index < 3 else index - 3, p[0], p[1]] = 1.0 - i * 0.03
        else:
            for i, p in enumerate(pos):
                obs[3, p[0], p[1]] = 1.0 - i * 0.03

    for bean in state[1]:
        obs[4, bean[0], bean[1]] = 1.0
    
    obs[:, -2:, :] = obs[:, 0:2, :]
    obs[:, :, -2:] = obs[:, :, 0:2]
    
    return obs

class Game(AbstractGame):
    def __init__(self, seed=None):
        file_path = os.path.join(os.path.dirname(__file__), '_snakes/config.json')
        with open(file_path) as f:
            conf = json.load(f)['snakes_3v3']
        class_literal = conf['class_literal']
        self.env = getattr(_snakes, class_literal)(conf)
        if seed is not None:
            torch.manual_seed(1)
            np.random.seed(1)
            random.seed(1)
        self.action_1p = None
        self.obs = None
        self.onehot_actions = np.empty((4,), dtype=object)
        self.onehot_actions[:] = np.eye(4, dtype=int).tolist()

    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        if self.action_1p != None:
            action = self.onehot_actions[np.concatenate((get_vec_action(self.action_1p), get_vec_action(action)))[:, np.newaxis]]
            self.obs, reward, done, _, _ = self.env.step(action)
            self.action_1p = None
            return [get_observation(self.obs[0]), get_observation(self.obs[3])], sum(reward[3:]) - sum(reward[:3]), done
        else:
            self.action_1p = action
            return [None, None], 0.0, False

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return 0 if self.action_1p is None else 1

    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.
        
        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.        

        Returns:
            An array of integers, subset of the action space.
        """
        return list(range(64))

    def reset(self):
        """
        Reset the game for a new game.
        
        Returns:
            Initial observation of the game.
        """
        self.obs = self.env.reset()
        return [get_observation(self.obs[0]), get_observation(self.obs[3])]

    def close(self):
        """
        Properly close the game.
        """
        pass

    def render(self):
        """
        Display the game observation.
        """
        pass

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.

        Returns:
            String representing the action.
        """
        actions = {
            0: "Up",
            1: "Down",
            2: "Left",
            3: "Right",
        }
        vec_act = get_vec_action(action_number)
        return f"{action_number}. {actions[vec_act[0]]} {actions[vec_act[1]]} {actions[vec_act[2]]}"
