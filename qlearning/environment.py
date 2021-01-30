from typing import Any, Tuple

import numpy as np
from tensorflow.keras import Model


class DQNEnvironment:
    def __init__(self) -> None:
        """
        Initialise your environment here. Any basic variables and other
        structures should be declared here.
        """
        pass

    def create_model(self, learning_rate=0.0001) -> Model:
        """
        Return a model to use with this environment.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate to pass to the optimizer, by default 0.0001

        Returns
        -------
        Model
            The new Keras model
        """
        raise NotImplementedError()
    
    def step(self, action, memory_increment=0) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take a step in the environment using the given action.

        Parameters
        ----------
        action : int
            The action that the neural network has decided to take. This may
            either be a value returned from the actual Keras model, or it may
            be a value returned from this own class' `get_sample_action`.
        memory_increment : int, optional
            This value refers to the ID of the memory that this action will create.
            This could be used to keep track of the memory that is created, and
            reward this behaviour at a later time.

            Training efficiency may vary depending how much later a reward is
            issued, as the DQN may have already used that memory one or more
            times to train.

        Returns
        -------
        Tuple[np.ndarray, float, bool, dict]
            The return value is a tuple of 4 values:
            - The new game state. This should return values in the same "format"
              as the `game_state` method on this class. 
            - The reward value for the action
            - Whether this round of training is complete
            - A dictionary of any "debug" information that may be used by the
              training loop.
        """
        raise NotImplementedError()

    def get_sample_action(self) -> int:
        """
        Get a "random" action to be used by the DQN to aid training. This is
        used less an less as the DQN instance's epsilon value decays.

        Returns
        -------
        int
            The random action
        """
        raise NotImplementedError()
    
    def game_state(self) -> np.ndarray:
        """
        Return the current game state as a `np.ndarray` with dimensions (1, X)
        where X is the input dimension of the model returned by `create_model`.

        Returns
        -------
        np.ndarray
            The game state preapred for the Model.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Reset the current game state to what it would be at the start of a
        game. This should discard any round specific information and reset
        the environment to be used again for training.
        """
        raise NotImplementedError()
