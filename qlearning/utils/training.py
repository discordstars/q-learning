from collections import deque
from functools import wraps
from typing import Generator, Optional, Tuple

import numpy as np

from ..dqn import DQN
from ..environment import DQNEnvironment


def iter_train(
    environment: DQNEnvironment,
    dqn: DQN,
    trials: int = 5000,
    step_train_interval: Optional[int] = 200,
    train_batch_size: int = 32,
    yield_on_step: bool = False,
) -> Generator[Tuple[int, int, Tuple[np.ndarray, float, bool, dict]]]:
    """
    An all-purpose DQN training generator. This performs the following
    procedure `trial` times.

    This resets the environment, and gets the current game state. Until the
    `DQNEnvironment.step` method returns `done` as ``True``, it gets the
    DQN to act on the current state, uses the action in the step function and
    then trains if `step_train_interval` is not ``None`` and the interval
    has been reached.

    This looks at the ``"reward_updates"`` key in the `step` function's data
    and passes every dict in the list as ``**kwargs`` to the DQN's
    `update_reward` function.
    

    Parameters
    ----------
    environment : DQNEnvironment
        The environment to train in.
    dqn : DQN
        The DQN to train
    trials : int, optional
        The number of trials/games, by default 5000
    step_train_interval : Optional[int], optional
        The number of steps to take before training, by default 200
    train_batch_size : int, optional
        Number of memories to train with each time, by default 32
    yield_on_step : bool, optional
        Whether to yield on each step, by default False

    Yields
    -------
    Tuple[int, int, Tuple[np.ndarray, float, bool, dict]]
        The trial #, step # and most recent memory (if end of trial, memory is
        an empty list)
    """
    for trial in range(trials):
        # Reset the game environment and get the new state
        environment.reset()
        cur_state = environment.game_state()

        # Whether this round is complete
        done = False
        steps = 0
        while not done:
            # Perform current action
            action = dqn.act(cur_state)
            new_state, reward, done, data = environment.step(
                action, memory_increment=dqn.memory_increment + 1
            )

            # Allows the step function to update rewards
            for reward_update in data.get("reward_updates", []):
                dqn.update_reward(**reward_update)

            # Store the new memory
            _, memory = dqn.store_memory(cur_state, action, reward, new_state, done)

            # Yield trial #, step # and last memory
            if yield_on_step:
                yield trial, steps, memory

            # If step training interval is not none and we're at that interval,
            # learn from given batch size
            if step_train_interval and steps % step_train_interval == 0:
                dqn.replay_memory(batch_size=train_batch_size)
                dqn.target_train()

        # Yield trial #, total step count for this trial and empty memory
        yield trial, steps, []

        # Learn from given batch size
        dqn.replay_memory(batch_size=train_batch_size)
        dqn.target_train()


@wraps(iter_train)
def train(
    environment: DQNEnvironment,
    dqn: DQN,
    trials: int = 5000,
    step_train_interval: Optional[int] = 200,
    train_batch_size: int = 32,
):
    """
    This function wraps `iter_train`, but doesn't yield. Instead it uses a
    `deque` of size ``0`` to exhaust the generator.

    All parameters are passed to `iter_train`.

    Parameters
    ----------
    environment : DQNEnvironment
        The environment to train in.
    dqn : DQN
        The DQN to train
    trials : int, optional
        The number of trials/games, by default 5000
    step_train_interval : Optional[int], optional
        The number of steps to take before training, by default 200
    train_batch_size : int, optional
        Number of memories to train with each time, by default 32
    """
    # Use empty deque to exhaust generator without caring about return values
    deque(
        iter_train(
            environment=environment,
            dqn=dqn,
            trials=trials,
            step_train_interval=step_train_interval,
            train_batch_size=train_batch_size,
        ),
        maxlen=0,
    )
