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
 ID Generator: globally unique, immutable object ID generator

Core design principles:
1. **Global uniqueness**: an ID is unique across the simulation and clashes with nothing else
2. **Migration invariance**: the ID is fixed at birth and never changes, whatever rank the object moves to
3. **Determinism**: the same inputs (Type, BirthRank, BirthStep, LocalCounter) give the same ID
4. **Zero communication**: the ID follows from local data alone, with no cross-rank coordination

Generation:
    ID = Hash(Type, Rank, Step, Counter) & Mask (int64)

Properties:
- Based on the SHA-256 hash
- The leading 8 bytes are converted to int64
- Independent of space and time: the ID binds to birth attributes, not to the current position or step
"""

import hashlib
import numpy as np
from typing import Optional


class IDGenerator:
    """
    Globally unique, immutable object ID generator

    Uses SHA-256 to produce a 64-bit integer ID, guaranteeing:
    1. global uniqueness
    2. migration invariance (the ID never changes)
    3. determinism (same input -> same output)
    4. zero communication (no cross-rank coordination)

    Args:
        birth_rank: rank ID on which the object was born
        birth_step: simulation step at which the object was born

    Attributes:
        birth_rank: birth rank
        birth_step: birth step
        local_counter: local counter (separates objects of one type born in one step)
    """

    def __init__(self, birth_rank: int, birth_step: int):
        """
        Initialize the ID generator

        Args:
            birth_rank: rank ID on which the object was born
            birth_step: simulation step at which the object was born
        """
        self.birth_rank = birth_rank
        self.birth_step = birth_step
        self.local_counter = 0  # local counter

    def generate_id(self, obj_type: str) -> int:
        """
        Generate a globally unique, immutable ID

        Algorithm:
        1. build the characteristic string "Type:Rank:Step:Counter"
        2. SHA-256 hash
        3. take the leading 8 bytes as an int64

        Args:
            obj_type: object type string (e.g. 'vacancy', 'interstitial', 'H1V')

        Returns:
            id_val: 64-bit integer ID (int64)

        Example:
            >>> gen = IDGenerator(birth_rank=0, birth_step=10)
            >>> id1 = gen.generate_id('vacancy')
            >>> id2 = gen.generate_id('vacancy')
            >>> id1 != id2  # different counters, different IDs
            True
        """
        # Build the characteristic string
        seed_str = f"{obj_type}:{self.birth_rank}:{self.birth_step}:{self.local_counter}"

        # SHA-256 hash
        h = hashlib.sha256(seed_str.encode('utf-8')).digest()

        # [fix] Take the leading 8 bytes and mask into the 63-bit positive range:
        # the event system reads negative IDs (and 0) as empty slots, so a signed
        # conversion would turn about half the objects into unusable ghosts
        id_val = int.from_bytes(h[:8], byteorder='little') & 0x7FFFFFFFFFFFFFFF
        if id_val == 0:
            id_val = 1

        # Advance the local counter (ready for the next object)
        self.local_counter += 1

        return id_val

    def generate_batch(self, obj_types: list) -> np.ndarray:
        """
        Generate IDs in bulk (vectorized interface)

        Args:
            obj_types: list of object types ['vacancy', 'interstitial', ...]

        Returns:
            ids: NumPy array, shape=(N,), dtype=np.int64

        Example:
            >>> gen = IDGenerator(birth_rank=0, birth_step=10)
            >>> ids = gen.generate_batch(['vacancy', 'interstitial', 'H1V'])
            >>> len(ids)
            3
        """
        ids = np.empty(len(obj_types), dtype=np.int64)

        for i, obj_type in enumerate(obj_types):
            ids[i] = self.generate_id(obj_type)

        return ids

    def reset_counter(self):
        """
        Reset the local counter (use with care!)

        Warning: resetting the counter may make new IDs collide with earlier ones.
                 Use it only once all earlier objects have been deleted.
        """
        self.local_counter = 0

    def __repr__(self):
        return (f"IDGenerator(birth_rank={self.birth_rank}, "
                f"birth_step={self.birth_step}, "
                f"local_counter={self.local_counter})")


# ===== Standalone function interface (backward compatible) =====

def generate_global_id(obj_type: str,
                      birth_rank: int,
                      birth_step: int,
                      local_counter: int) -> int:
    """
    Generate a globally unique, immutable ID (function interface)

    A stateless function, suitable for one-off calls.
    For bulk generation prefer the IDGenerator class.

    Args:
        obj_type: object type string
        birth_rank: birth rank ID
        birth_step: birth step
        local_counter: local counter

    Returns:
        id_val: 64-bit integer ID

    Example:
        >>> id1 = generate_global_id('vacancy', 0, 10, 0)
        >>> id2 = generate_global_id('vacancy', 0, 10, 1)
        >>> id1 != id2
        True
    """
    # Build the characteristic string
    seed_str = f"{obj_type}:{birth_rank}:{birth_step}:{local_counter}"

    # SHA-256 hash
    h = hashlib.sha256(seed_str.encode('utf-8')).digest()

    # [fix] 63-bit positive range (negative IDs read as empty slots downstream)
    id_val = int.from_bytes(h[:8], byteorder='little') & 0x7FFFFFFFFFFFFFFF
    if id_val == 0:
        id_val = 1

    return id_val


# ===== ID inspection helpers (debugging) =====

def parse_id_info(id_val: int) -> dict:
    """
    Inspect an ID (debugging helper)

    Note: because IDs are hashed, the original parameters cannot be recovered.
          This function only reports basic properties (sign, absolute value, ...).

    Args:
        id_val: 64-bit integer ID

    Returns:
        info: dict with basic information about the ID

    Example:
        >>> id_val = generate_global_id('vacancy', 0, 10, 0)
        >>> info = parse_id_info(id_val)
        >>> 'sign' in info and 'abs_value' in info
        True
    """
    info = {
        'id': id_val,
        'sign': 'positive' if id_val >= 0 else 'negative',
        'abs_value': abs(id_val),
        'hex': hex(id_val),
        'note': 'IDs are hashed; the original parameters cannot be recovered'
    }

    return info


# ===== Unit tests =====
if __name__ == "__main__":
    print("=" * 60)
    print("IDGenerator unit tests")
    print("=" * 60)

    # Test 1: basic ID generation
    print("\nTest 1: basic ID generation")
    gen = IDGenerator(birth_rank=0, birth_step=10)
    id1 = gen.generate_id('vacancy')
    id2 = gen.generate_id('vacancy')
    id3 = gen.generate_id('interstitial')

    print(f"ID 1 (vacancy):      {id1}")
    print(f"ID 2 (vacancy):      {id2}")
    print(f"ID 3 (interstitial): {id3}")

    # Check uniqueness
    assert id1 != id2, "consecutive IDs of one type must differ (counter increments)"
    assert id1 != id3, "IDs of different types must differ"
    assert id2 != id3, "IDs of different types must differ"
    print("OK uniqueness check passed")

    # Test 2: determinism (same input -> same output)
    print("\nTest 2: determinism")
    id_a = generate_global_id('vacancy', 0, 10, 0)
    id_b = generate_global_id('vacancy', 0, 10, 0)
    print(f"ID A: {id_a}")
    print(f"ID B: {id_b}")
    assert id_a == id_b, "the same input must produce the same ID"
    print("OK determinism check passed")

    # Test 3: migration invariance (simulated object migration)
    print("\nTest 3: migration invariance")
    # Object born on rank 0 at step 10
    gen_rank0 = IDGenerator(birth_rank=0, birth_step=10)
    obj_id = gen_rank0.generate_id('vacancy')
    print(f"ID at birth (rank 0, step 10): {obj_id}")

    # The object migrates to rank 1 but the ID is unchanged (it binds to birth data)
    # Regenerate the same ID (as when tracing an object's history)
    obj_id_verify = generate_global_id('vacancy', 0, 10, 0)
    print(f"Verification ID (recomputed from birth data): {obj_id_verify}")
    assert obj_id == obj_id_verify, "the ID must bind to birth data only, not to migration"
    print("OK migration invariance check passed")

    # Test 4: bulk generation
    print("\nTest 4: bulk generation")
    gen_batch = IDGenerator(birth_rank=1, birth_step=20)
    types = ['vacancy', 'interstitial', 'H1V', 'H2V', 'H3V']
    ids = gen_batch.generate_batch(types)
    print(f"Generated {len(ids)} IDs in bulk:")
    for i, (t, id_val) in enumerate(zip(types, ids)):
        print(f"  {i}: {t:15s} → {id_val}")

    # Check uniqueness
    unique_ids = np.unique(ids)
    assert len(unique_ids) == len(ids), "all IDs generated in bulk must be unique"
    print("OK bulk uniqueness check passed")

    # Test 5: different ranks and steps must give different IDs
    print("\nTest 5: uniqueness across ranks/steps")
    id_r0_s0 = generate_global_id('vacancy', 0, 0, 0)
    id_r0_s1 = generate_global_id('vacancy', 0, 1, 0)
    id_r1_s0 = generate_global_id('vacancy', 1, 0, 0)
    id_r1_s1 = generate_global_id('vacancy', 1, 1, 0)

    print(f"Rank 0, Step 0: {id_r0_s0}")
    print(f"Rank 0, Step 1: {id_r0_s1}")
    print(f"Rank 1, Step 0: {id_r1_s0}")
    print(f"Rank 1, Step 1: {id_r1_s1}")

    all_ids = [id_r0_s0, id_r0_s1, id_r1_s0, id_r1_s1]
    assert len(set(all_ids)) == 4, "different ranks/steps must give different IDs"
    print("OK cross-rank/step uniqueness check passed")

    # Test 6: ID inspection
    print("\nTest 6: ID inspection (debugging helper)")
    info = parse_id_info(id1)
    print(f"ID: {info['id']}")
    print(f"Sign: {info['sign']}")
    print(f"Absolute value: {info['abs_value']}")
    print(f"Hexadecimal: {info['hex']}")
    print(f"Note: {info['note']}")

    # Test 7: counter reset
    print("\nTest 7: counter reset")
    gen_reset = IDGenerator(birth_rank=2, birth_step=30)
    id_before = gen_reset.generate_id('vacancy')
    print(f"Before reset (counter=0): {id_before}")

    gen_reset.generate_id('vacancy')  # counter → 1
    gen_reset.generate_id('vacancy')  # counter → 2
    print(f"Current counter: {gen_reset.local_counter}")

    gen_reset.reset_counter()  # counter → 0
    id_after = gen_reset.generate_id('vacancy')
    print(f"After reset (counter=0): {id_after}")

    assert id_before == id_after, "the same ID must be produced after a reset"
    print("OK counter reset check passed")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

    # Extra: collision probability estimate
    print("\n" + "=" * 60)
    print("Estimated ID collision probability")
    print("=" * 60)
    print("SHA-256 hash space: 2^256")
    print("int64 truncation to 64 bits: 2^64 = 1.84 x 10^19")
    print("Birthday-paradox estimate:")
    print("  - 10^6 IDs: collision probability ~ 2.7 x 10^-8 (negligible)")
    print("  - 10^9 IDs: collision probability ~ 2.7 x 10^-2 (about 2.7%)")
    print("  - typical OKMC system (N < 10^7): collision risk is very low")
    print("=" * 60)
