import random
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .environment import DQNEnvironment


@dataclass
class DQNParams:
    """
    Describes parameters to be used by the DQN.

    `epsilon`
        The possibility that the DQN will use a random value provided
        by the environment via `get_sample_action`.
    `epsilon_min`
        The minimum value of `epsilon`.
    `epsilon_decay`
        The rate of decay for `epsilon`, where `epsilon *= epsilon_decay`for
        each call to `DQN.act`.
    `gamma`
        The discount factor of the DQN. Models the fact that the DQN is unsure
        whether its action will result in the world "ending"
        (`DQNEnvironment.reset` being called).

        Useful read: https://stats.stackexchange.com/a/221472.
    `learning_rate`
        The learning rate of the model. This is solely stored for the purpose
        of passing it to `DQNEnvironment.create_model`.
    """

    # Chance of using random sample "exploration" instead of using data
    # "exploitation"
    epsilon: float = 1.0
    epsilon_min: float = 0.01  # The minimum value for epsilon
    epsilon_decay: float = 0.995  # The reduction of epsilon
    gamma: float = 0.85  # Discount factor
    learning_rate: float = 0.005  # Learning rate


class DQN:
    # Basic concept designed around the following article
    # https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    def __init__(
        self,
        environment: DQNEnvironment,
        memory_size: int = 2000,
        params: DQNParams = DQNParams(),
    ) -> None:
        """
        Initialize a new DQN for an environment.

        Parameters
        ----------
        environment : DQNEnvironment
            The environment that the DQN exists in and will train inside.

            The environment must be an instance of a subclass of `DQNEnvironment`,
            otherwise it will not function correctly.
        memory_size : int, optional
            The amount of "memories" (action -> result records), by default 2000

            The higher this is, the more actions the DQN will store and can be
            used in training.
        params : DQNParams, optional
            The parameters for the training environment, by default DQNParams()

            The params must be an instance of DQNParams.
        """
        if not isinstance(environment, DQNEnvironment):
            raise ValueError(
                "environment must be of a subclass of DQNEnvironment, not {}".format(
                    type(environment)
                )
            )
        elif not isinstance(params, DQNParams):
            raise ValueError(
                "params must be an instance of DQNParams, not {}".format(type(params))
            )

        self.env = environment
        self.memory = deque(maxlen=memory_size)
        self.memory_increment = 0

        self.epsilon = params.epsilon
        self.epsilon_min = params.epsilon_min
        self.epsilon_decay = params.epsilon_decay
        self.gamma = params.gamma
        self.learning_rate = params.learning_rate

        # Create a model and a target model.
        # - `model` runs the predictions for our AI in this environment, and
        #   is what we want to train.
        # - `target_model` tracks the action we want our model to take.
        #
        # This is a "hack" figured out by DeepMind to help achieve convergence
        self.model = self.env.create_model(self.learning_rate)
        self.target_model = self.env.create_model(self.learning_rate)

    def store_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        new_state: np.ndarray,
        done: bool,
    ) -> Tuple[int, Tuple[np.ndarray, float, bool, dict]]:
        """
        Store this memory and return the memory increment that refers to it

        Parameters
        ----------
        state : np.ndarray
            The state of the game before the action was taken
        action : int
            The returned action
        reward : float
            The reward gained for this action
        new_state : np.ndarray
            The state of the game after the action was taken
        done : bool
            Whether this action resulted in the game "finishing"

        Returns
        -------
        int
            The ID of the memory (can be used with `update_reward`)
        """
        self.memory_increment += 1
        memory = [state, action, reward, new_state, done, self.memory_increment]
        self.memory.append(memory)
        return memory[-1], memory[:-1]

    def update_reward(
        self, memory_id: int, reward: float, increment: bool = False
    ) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Update the reward value for a previous memory.

        NOTE: If the memory can not be found, this will do nothing.

        Parameters
        ----------
        memory_id : int
            The ID of the memory returned by `store_memory`.
        reward : float
            The reward value to either add to (if `increment` is ``True``) or
            replace (if `increment` is ``False``) the current reward value.
        increment : bool, optional
            Whether to add to the existing value, by default False

            If False, the `reward` parameter will become the new reward value
            for the found memory.

        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            The new memory (with memory updated)
        """
        for memory in self.memory:
            _, _, reward, _, _, memory_increment = memory
            if memory_increment == memory_id:
                if increment:
                    memory[2] = memory[2] + reward
                else:
                    memory[2] = reward

                # Don't return the memory increment
                return memory[:-1]

    def replay_memory(self, batch_size: int = 32):
        """
        Train the model on `batch_size` memories, using the target model to
        predict the reward if the memory isn't an end state.

        In non-terminal states, it uses the discount factor for any future
        reward estimates and adds the reward of the round plus the
        Q_future reward as the new estimated reward.

        Parameters
        ----------
        batch_size : int, optional
            Number of memories to use, by default 32
        """
        if len(self.memory) < batch_size:
            # Don't train if memory too small
            return

        # Get a random batch of memories to use to train
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            # Discard the memory increment value
            state, action, reward, new_state, done, _ = sample
            # The action taken by our target parameter
            target = self.target_model.predict(state)
            if done:
                # If a terminal state, this is the ultimate reward
                target[0][action] = reward
            else:
                # In a non terminal state, find the maximum reward for any
                # possible future state. This is because the current action may
                # provide a low reward, but consequently possible actions can
                # provide higher rewards.
                Q_future = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            # Train for one epoch
            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        """
        Replace the target model's weights by the current model's weights.

        This should be used alongside `replay_memory`.
        """
        weights = self.model.get_weights()
        self.target_model.set_weights(weights.copy())

    def act(
        self,
        state: np.ndarray,
        decay_epsilon: bool = True,
        always_use_model: bool = False,
    ) -> int:
        """
        Return an action based on the given game state, either from the
        model or from the environments `get_sample_action` function.

        Parameters
        ----------
        state : np.ndarray
            The current game state.
        decay_epsilon : bool, optional
            Whether to decay epsilon, by default True
        always_use_model : bool, optional
            Whether to always use the model , by default False

            If ``True``, the value of `epsilon` is ignored in decision making.

        Returns
        -------
        int
            The action taken by this DQN.
        """
        # Decay of dependence of random sample data
        if decay_epsilon:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        if not always_use_model and np.random.random() < self.epsilon:
            # Use sample data
            return self.env.get_sample_action()

        return np.argmax(self.model(state)[0])

    def save_model(self, fn: str):
        """
        Save this model to the given path.

        Parameters
        ----------
        fn : str
            Filename passed to `Model.save`
        """
        self.model.save(fn)
