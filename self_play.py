import math
import time

import numpy
import ray
import torch
import random

import models


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)
        self.mcts = SM_MCTS(config) if config.simultaneous else MCTS(config)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            if not test_mode:
                game_history = self.play_game(
                    self.config.visit_softmax_temperature_fn(
                        trained_steps=ray.get(
                            shared_storage.get_info.remote("training_step")
                        )
                    ),
                    self.config.temperature_threshold,
                    False,
                    "self",
                    0,
                )

                replay_buffer.save_game.remote(game_history, shared_storage)

            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    0,
                    self.config.temperature_threshold,
                    False,
                    "self" if len(self.config.players) == 1 else self.config.opponent,
                    self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(
                            reward if game_history.to_play_history[i - 1] == self.config.muzero_player else -reward
                            for i, reward in enumerate(game_history.reward_history)
                        ),
                        "mean_value": numpy.mean(
                            [value if game_history.to_play_history[i - 1] == self.config.muzero_player else -value
                            for i, value in enumerate(game_history.root_values) if value]
                        ),
                    }
                )
                if 1 < len(self.config.players):
                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": sum(
                                reward if game_history.to_play_history[i - 1] == self.config.muzero_player else -reward
                                for i, reward in enumerate(game_history.reward_history)
                            ),
                            "opponent_reward": sum(
                                reward if game_history.to_play_history[i - 1] != self.config.muzero_player else -reward
                                for i, reward in enumerate(game_history.reward_history)
                            ),
                        }
                    )

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game()

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        simultaneous = self.config.simultaneous

        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append([0 for _ in range(len(self.config.players))] if simultaneous else 0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                if simultaneous:
                    assert (
                        len(observation) == len(self.config.players)
                    ), f"When N agents act simultaneously, the environment should return an tuple of N observations."
                    for obs in observation:
                        assert (
                            len(numpy.array(obs).shape) == 3
                        ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(obs).shape)} dimensionnal. Got observation of shape: {numpy.array(obs).shape}"
                        assert (
                            numpy.array(obs).shape == self.config.observation_shape
                        ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(obs).shape}."
                else:
                    assert (
                        len(numpy.array(observation).shape) == 3
                    ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                    assert (
                        numpy.array(observation).shape == self.config.observation_shape
                    ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."

                stacked_observations = game_history.get_stacked_observations(
                    -1,
                    self.config.stacked_observations,
                    self.config.observation_shape,
                    simultaneous,
                    len(self.config.players),
                    len(self.config.action_space),
                )

                # Choose the action
                if simultaneous:
                    action, root = self.select_mcts_action(
                        stacked_observations,
                        self.game.to_play(),
                        self.game.legal_actions(),
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                        render,
                    )
                    if opponent != "self":
                        for player in self.config.players:
                            if player != muzero_player:
                                action[player] = self.select_opponent_action(
                                    opponent, stacked_observations[player], player, simultaneous,
                                )
                else:
                    player = self.game.to_play()
                    if opponent == "self" or muzero_player == self.game.to_play():
                        action, root = self.select_mcts_action(
                            stacked_observations,
                            player,
                            self.game.legal_actions(),
                            temperature
                            if not temperature_threshold
                            or len(game_history.action_history) < temperature_threshold
                            else 0,
                            render,
                        )
                    else:
                        action = self.select_opponent_action(
                            opponent, stacked_observations, player
                        )
                        root = None

                observation, reward, done = self.game.step(action)

                if render:
                    if simultaneous:
                        for player in self.config.players:
                            print(f"Player {player} played action: {self.game.action_to_string(action[player])}")
                    else:
                        print(f"Player {player} played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_history

    def close_game(self):
        self.game.close()
    
    def select_mcts_action(self, stacked_observations, player, legal_actions, temperature, render):
        root, mcts_info = self.mcts.run(
            self.model,
            stacked_observations,
            legal_actions,
            player,
            True,
        )
        if isinstance(root, DecoupledNode):
            action = [self.select_action(subnode, temperature) for subnode in root.subnodes]
        else:
            action = self.select_action(
                root,
                temperature
            )

        if render:
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(
                f"Root value for player {player}: {root.value():.2f}"
            )
        
        return action, root

    def select_opponent_action(self, opponent, stacked_observations, to_play, simultaneous=False):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = self.mcts.run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                to_play,
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {to_play}: {root.value():.2f}")
            if simultaneous:
                suggested_action = self.select_action(root.subnodes[to_play], 0)
            else:
                suggested_action = self.select_action(root, 0)
            print(
                f"Player {to_play} turn. MuZero suggests {self.game.action_to_string(suggested_action)}"
            )
            return self.game.human_to_action()
        elif opponent == "expert":
            return self.game.expert_agent()
        elif opponent == "random":
            legal_actions = self.game.legal_actions()[to_play] if simultaneous else self.game.legal_actions()
            assert (legal_actions), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(set(self.config.action_space)), "Legal actions should be a subset of the action space."

            return select_random(legal_actions)
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        if temperature == 0:
            action = node.actions[numpy.argmax(node.visit_counts)]
        elif temperature == float("inf"):
            action = select_random(node.actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = node.visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / torch.sum(visit_count_distribution)
            # action = numpy.random.choice(node.actions, p=visit_count_distribution)
            action = node.actions[torch.multinomial(visit_count_distribution, 1)]

        return action

def select_random(list_):
    """ 
    Random selection of an element from a list.
    Faster than numpy.random.choice
    
    """
    return list_[math.floor(len(list_) * random.random())] 

# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config
        self.K = - math.log(config.pb_c_base) + config.pb_c_init  # Constant for faster calculation of the UCB

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        assert (
            legal_actions
        ), f"Legal actions should not be an empty array. Got {legal_actions}."
        assert set(legal_actions).issubset(
            set(self.config.action_space)
        ), "Legal actions should be a subset of the action space."

        root = Node()
        observation = (
            torch.tensor(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
        )
        (
            root_predicted_value,
            reward,
            policy_logits,
            hidden_state,
        ) = model.initial_inference(observation)
        root_predicted_value = models.support_to_scalar(
            root_predicted_value, self.config.support_size
        ).item()
        reward = models.support_to_scalar(reward, self.config.support_size).item()

        if add_exploration_noise:
            policy_logits = self.add_exploration_noise(
                policy_logits,
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        root.expand(
            legal_actions,
            to_play,
            reward,
            policy_logits,
            hidden_state,
        )

        min_max_stats = MinMaxStats()

        n_player = len(self.config.players)
        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = []
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                ai = self.select_action(node, min_max_stats)
                search_path.append((node, ai))
                node = node.children[ai]

                # Players play turn by turn
                virtual_to_play = (virtual_to_play + 1) % n_player

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent, _ = search_path[-1]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[parent.actions[ai]]]).to(parent.hidden_state.device),
            )

            value = models.support_to_scalar(value, self.config.support_size).item()
            reward = models.support_to_scalar(reward, self.config.support_size).item()
            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
            )

            self.backpropagate(search_path, reward, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info
    
    def add_exploration_noise(self, priors, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        noise = torch.tensor(numpy.random.dirichlet([dirichlet_alpha] * len(priors)), device=priors.device)
        return priors * (1- exploration_fraction) + noise * exploration_fraction
    
    def select_action(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log1p(node.visit_count + self.config.pb_c_base)
            + self.K
        ) * math.sqrt(node.visit_count)
        actions_ucb_scores = pb_c * node.priors / (node.visit_counts + 1) + (min_max_stats.normalize(node.values))

        # ai = select_random(numpy.argwhere(actions_ucb_scores == numpy.max(actions_ucb_scores)).flatten())
        ai = torch.argmax(actions_ucb_scores)
        return ai

    def backpropagate(self, search_path, reward, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            value = reward + self.config.discount * value
            for node, action_index in reversed(search_path):
                updated_action_value = node.update(value, action_index)
                min_max_stats.update(updated_action_value)

                value = node.reward + self.config.discount * value

        elif len(self.config.players) == 2:
            value = reward - self.config.discount * value
            for node, action_index in reversed(search_path):
                updated_action_value = node.update(value, action_index)
                min_max_stats.update(updated_action_value)

                value = node.reward - self.config.discount * value
        else:
            raise NotImplementedError("More than two player mode not implemented.")

class SM_MCTS(MCTS):
    """
    Monte Carlo Tree Search algorithm for simultaneous move game.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """
    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
    ):
        """
        At the root of the search tree we use the representation function to obtain
        hidden states given agents' observations.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        The predicted value is aligned with the `to_play` player, but the actions of all players
        can be extracted.   
        """
        assert to_play == 0

        for _legal_actions in legal_actions:
            assert (
                _legal_actions
            ), f"Legal actions should not be an empty array. Got {_legal_actions}."
            assert set(_legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

        root = DecoupledNode()
        values, rewards, logits, hidden_states = [], [], [], []
        obs = (
            torch.tensor(observation)
            .float()
            .to(next(model.parameters()).device)
        )
        (
            values,
            rewards,
            logits,
            hidden_states,
        ) = model.initial_inference(obs)
        values = models.support_to_scalar(values, self.config.support_size)
        rewards = models.support_to_scalar(rewards, self.config.support_size)
        root_predicted_value = (values[0] - values[1]).item() / 2.0
        reward = (rewards[0] - rewards[1]).item() / 2.0

        if add_exploration_noise:
            for i, policy_logits in enumerate(logits):
                logits[i] = self.add_exploration_noise(
                    policy_logits,
                    dirichlet_alpha=self.config.root_dirichlet_alpha,
                    exploration_fraction=self.config.root_exploration_fraction,
                )

        root.expand(
            legal_actions,
            to_play,
            reward,
            logits,
            hidden_states,
        )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            node = root
            search_path = []
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                ais = []
                for subnode in node.subnodes:
                    ais.append(self.select_action(subnode, min_max_stats))

                search_path.append((node, ais))
                node = node.children[ais[0]][ais[1]]
            
            parent, _ = search_path[-1]
            action = [parent.subnodes[0].actions[ais[0]], parent.subnodes[1].actions[ais[1]]]
            action = torch.tensor([action, numpy.flip(action)])

            values, rewards, logits, hidden_states = model.recurrent_inference(
                parent.hidden_state,
                action.to(parent.hidden_state.device),
            )
            values = models.support_to_scalar(values, self.config.support_size)
            rewards = models.support_to_scalar(rewards, self.config.support_size)
            value = (values[0] - values[1]).item() / 2.0
            reward = (rewards[0] - rewards[1]).item() / 2.0

            node.expand(
                [self.config.action_space] * len(self.config.players),
                to_play,
                reward,
                logits,
                hidden_states,
            )

            value = reward + self.config.discount * value
            self.backpropagate(search_path, value, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def backpropagate(self, search_path, value, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        assert len(self.config.players) == 2
        for node, action_index in reversed(search_path):
            updated_action_values = node.update(value, action_index)
            for updated_action_value in updated_action_values:
                min_max_stats.update(updated_action_value)

            value = node.reward + self.config.discount * value


class Node:
    def __init__(self):
        self.visit_count = 0
        self.priors = None
        self.actions = None
        self.values = None
        self.visit_counts = None
        self.to_play = -1
        self.children = []
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        self.priors = torch.softmax(policy_logits[0, actions], dim=0)
        self.actions = actions
        self.actions = actions
        self.values = torch.zeros_like(self.priors)
        self.visit_counts = torch.zeros_like(self.priors, dtype=int)
        self.children = [Node() for _ in self.actions]
    
    def update(self, value, action_index):
        self.visit_count += 1
        self.visit_counts[action_index] += 1
        self.values[action_index] += (value - self.values[action_index]) / self.visit_counts[action_index]
        return self.values[action_index]
    
    def value(self):
        return 0.0 if self.visit_count == 0 else torch.sum(self.values * self.visit_counts).item() / self.visit_count


class DecoupledNode:
    def __init__(self):
        self.children = []
        self.hidden_state = None
        self.reward = 0
        self.to_play = -1
        self.subnodes = []

    def expanded(self):
        return len(self.children) > 0

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        self.subnodes = [self.SubNode(acts, to_play, logits) for acts, to_play, logits in zip(actions, range(len(actions)), policy_logits)]
        self.children = [[DecoupledNode() for _ in actions[1]] for _ in actions[0]]

    def update(self, value, action_index):
        return [subnode.update(value if i == self.to_play else -value, action_index[i]) for i, subnode in enumerate(self.subnodes)]
    
    def value(self):
        return numpy.mean(
            [
                subnode.value() if i == self.to_play else -subnode.value()
                for i, subnode in enumerate(self.subnodes)
            ]
        )
    
    @property
    def visit_count(self):
        return self.subnodes[0].visit_count

    class SubNode:
        def __init__(self, actions, to_play, policy_logits):
            self.visit_count = 0
            self.to_play = to_play
            self.priors = torch.softmax(policy_logits[actions], dim=0)
            self.actions = actions
            self.values = torch.zeros_like(self.priors)
            self.visit_counts = torch.zeros_like(self.priors, dtype=int)

        def update(self, value, action_index):
            self.visit_count += 1
            self.visit_counts[action_index] += 1
            self.values[action_index] += (value - self.values[action_index]) / self.visit_counts[action_index]
            return self.values[action_index]
            
        def value(self):
            return 0.0 if self.visit_count == 0 else torch.sum(self.values * self.visit_counts).item() / self.visit_count

class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            if isinstance(root, DecoupledNode):
                child_visits_list = []
                for subnode in root.subnodes:
                    child_visits = numpy.zeros(len(action_space))
                    for ai, a in enumerate(subnode.actions):
                        child_visits[a] = subnode.visit_counts[ai] / subnode.visit_count
                    child_visits_list.append(child_visits)
                self.child_visits.append(child_visits_list)
            else:
                child_visits = numpy.zeros(len(action_space))
                for ai, a in enumerate(root.actions):
                    child_visits[a] = root.visit_counts[ai] / root.visit_count
                self.child_visits.append(child_visits)
            self.root_values.append(root.value())
        else:
            self.child_visits.append(None)
            self.root_values.append(None)

    def get_stacked_observations(self, index, num_stacked_observations, observation_shape, simultaneous, n_player, action_space_size):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)
        if simultaneous:
            stacked_observations = []
            for player in range(n_player):
                histories = [self.observation_history[index][player].copy(),]

                for past_observation_index in reversed(
                    range(index - num_stacked_observations, index)
                ):
                    if 0 <= past_observation_index:
                        histories.append(
                            [
                                numpy.ones(observation_shape[1:])
                                * self.action_history[past_observation_index + 1][player] / action_space_size
                            ]
                        )
                        histories.append(
                            self.observation_history[past_observation_index][player],
                        )
                    else:
                        histories.append(
                            [numpy.zeros(observation_shape[1:])],
                        )
                        histories.append(
                            numpy.zeros(observation_shape),
                        )

                stacked_observations.append(numpy.concatenate(histories))

        else:
            histories = [self.observation_history[index].copy(),]

            for past_observation_index in reversed(
                range(index - num_stacked_observations, index)
            ):
                if 0 <= past_observation_index:
                    histories.append(
                        [
                            numpy.ones(observation_shape[1:])
                            * self.action_history[past_observation_index + 1] / action_space_size
                        ]
                    )
                    histories.append(
                        self.observation_history[past_observation_index],
                    )
                else:
                    histories.append(
                        [numpy.zeros(observation_shape[1:])],
                    )
                    histories.append(
                        numpy.zeros(observation_shape),
                    )

            stacked_observations = numpy.concatenate(histories)

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
