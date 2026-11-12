# Copyright (c) 2026 Beihang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Deterministic random number generator (globally consistent)

Provides unified random number management across ranks, guaranteeing:
1. Reproducibility: the same user_seed yields the same random sequence
2. Global consistency: all ranks use the same random sequence
3. Determinism: enforced by the (user_seed, step, state) triple

Algorithm (counter-based construction of independent streams):
    Every random number is uniquely determined by the counter (user_seed,
    step, state); no state is carried between calls. The triple is passed as
    an entropy sequence to NumPy's SeedSequence (whose internal hashing makes
    distinct integer sequences yield statistically independent streams), which
    seeds a PCG64 generator. Ranks sharing a counter are synchronized already.

    References:
    - PCG64: M.E. O'Neill, "PCG: A Family of Simple Fast Space-Efficient
      Statistically Good Algorithms for Random Number Generation",
      HMC-CS-2014-0905 (2014)
    - NumPy SeedSequence spawning / parallel random number generation
      (Harris et al., Nature 585, 357–362, 2020)

State convention:
- state=1: engine internal - event selection (r1)
- state=2: engine internal - time advance (r2)
- state>=3: user calls - incremented on every call

Use cases:
- Engine internal: selecting events, computing the time step
- User side: random numbers inside execute functions (via config['rng'])
"""

import numpy as np
from typing import Union, List, Any


class RandomGenerator:
    """
    Deterministic random number generator (globally consistent)

    All ranks use the same random sequence, derived from (user_seed, step,
    state). Ranks stay in sync without any extra broadcast.

    Args:
        user_seed: global seed supplied by the user
        rank: rank of this process (kept for compatibility; does not affect output)

    Example:
        >>> rng = RandomGenerator(user_seed=42, rank=0)
        >>> rng.set_step(0)  # set the current step
        >>> r1 = rng.random()  # draw a random number in [0, 1)
        >>> idx = rng.randint(0, 10)  # draw a random integer in [0, 10)
        >>> direction = rng.choice([-1, 1])  # pick an element at random
    """

    def __init__(self, user_seed: int, rank: int):
        """
        Initialize the random number generator

        Args:
            user_seed: global seed supplied by the user
            rank: rank of this process (kept, but unused)
        """
        self.user_seed = user_seed
        self.rank = rank  # kept for compatibility
        self.current_step = 0
        self.user_call_counter = 0  # counter of user calls

    def set_step(self, step: int):
        """
        Set the current time step

        Args:
            step: current time step
        """
        self.current_step = step
        self.user_call_counter = 0  # reset the user call counter each step

    def _make_rng(self, state: int) -> np.random.Generator:
        """
        Build an independent stream from the (user_seed, step, state) counter

        [fix] The triple is handed to SeedSequence as an entropy sequence rather
        than packed into bit fields by XOR: the packing scheme aliases streams
        once state overflows 16 bits (>65533 user calls in one step), or for
        structured seeds. Tuple entropy has no fields, so triples map 1:1 to streams.

        Args:
            state: state selector
                1 = event selection (r1)
                2 = time advance (r2)
                >=3 = user calls
        """
        entropy = (self.user_seed & 0xFFFFFFFFFFFFFFFF,
                   self.current_step, state)
        return np.random.default_rng(entropy)

    def random(self) -> float:
        """
        Draw a random float in [0, 1)

        Returns:
            float: a random number in [0, 1)
        """
        self.user_call_counter += 1
        state = 2 + self.user_call_counter  # state >= 3 marks a user call
        return float(self._make_rng(state).random())

    def uniform(self, low: float, high: float) -> float:
        """
        Draw a uniform random float in [low, high)

        Args:
            low: lower bound (inclusive)
            high: upper bound (exclusive)

        Returns:
            float: a random number in [low, high)
        """
        self.user_call_counter += 1
        state = 2 + self.user_call_counter
        return float(self._make_rng(state).uniform(low, high))

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
        """
        Draw normally distributed random numbers

        Args:
            loc: mean
            scale: standard deviation
            size: output shape (None returns a scalar)

        Returns:
            float or np.ndarray
        """
        self.user_call_counter += 1
        state = 2 + self.user_call_counter
        result = self._make_rng(state).normal(loc, scale, size)
        return float(result) if size is None else result

    def randint(self, low: int, high: int) -> int:
        """
        Draw a random integer in [low, high)

        Args:
            low: lower bound (inclusive)
            high: upper bound (exclusive)

        Returns:
            int: a random integer in [low, high)
        """
        self.user_call_counter += 1
        state = 2 + self.user_call_counter
        return int(self._make_rng(state).integers(low, high))

    def choice(self, choices: Union[List[Any], np.ndarray]) -> Any:
        """
        Pick one element at random from the given choices

        Args:
            choices: list or array of candidate elements

        Returns:
            Any: the element chosen at random
        """
        self.user_call_counter += 1
        state = 2 + self.user_call_counter
        return self._make_rng(state).choice(choices)

    def random_for_event_selection(self) -> float:
        """
        Engine internal: draw the random number for event selection (r1)

        Returns:
            float: a random number in [0, 1)
        """
        return float(self._make_rng(1).random())

    def random_for_time_step(self) -> float:
        """
        Engine internal: draw the random number for the time advance (r2)

        Returns:
            float: a random number in [0, 1)
        """
        return float(self._make_rng(2).random())

    def reset_counter(self):
        """
        Reset the user call counter (invoked automatically each step)
        """
        self.user_call_counter = 0
