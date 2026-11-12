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
Incremental update manager

Core responsibilities:
1. Manage the incremental update strategy of the three event classes
2. Track changed objects
3. Track global state changes
4. Synchronize changed objects over MPI

Incremental update rules:
- zero-object events: check for global state changes (temperature, time, fields, ...)
- single-object events: check whether the object is in changed_objects
- pair events: always rebuilt in full (a KDTree spatial query is needed)

Switch:
- incremental update is on by default
- it can be disabled in the configuration (e.g. when coupling an external field)
"""

import numpy as np
from mpi4py import MPI
from typing import Set, List, Optional


class IncrementalUpdateManager:
    """
    Incremental update manager

    Tracks changed objects and decides whether events must be rebuilt
    """

    def __init__(self, comm: MPI.Comm,
                 enabled: bool = True,
                 verbose: bool = False):
        """
        Initialize the incremental update manager

        Args:
            comm: MPI communicator
            enabled: whether incremental update is enabled
            verbose: whether to print details
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.enabled = enabled
        self.verbose = verbose

        # Changed object tracking
        self.changed_objects: Set[int] = set()

        # Deleted object tracking (to clear ghost events next step)
        self.deleted_objects: Set[int] = set()

        # Coin persistence: migrated object tracking
        # Records obj_ids whose event is held here while the object lives elsewhere
        self.migrated_obj_ids: Set[int] = set()

        # Global state change tracking
        self.global_state_changed = False

        # User-forced rebuild flags (for external field changes and the like)
        self.force_rebuild_zero = False
        self.force_rebuild_single = False

        # First-step marker (the first step must be computed in full)
        self.first_step = True

        # Statistics
        self.stats = {
            'full_rebuilds': 0,
            'incremental_updates': 0,
            'skipped_zero_obj': 0,
            'skipped_single_obj': 0,
        }

    # ========================================
    # Change tracking interface
    # ========================================

    def mark_objects_changed(self, obj_ids: List[int]):
        """
        Mark objects as changed

        Args:
            obj_ids: list of changed object IDs
        """
        valid_ids = [oid for oid in obj_ids if oid >= 0]
        self.changed_objects.update(valid_ids)
        # Coin invalidation: a changed object voids the migrated event
        self.migrated_obj_ids.difference_update(valid_ids)

        if self.verbose and len(valid_ids) > 0:
            print(f"[Rank {self.rank}] Marked {len(valid_ids)} objects as changed")

    def mark_objects_deleted(self, obj_ids: List[int]):
        """
        Mark objects as deleted (to clear ghost events next step)

        Args:
            obj_ids: list of deleted object IDs
        """
        valid_ids = [oid for oid in obj_ids if oid >= 0]
        self.deleted_objects.update(valid_ids)
        # Remove from changed_objects (deleted objects need no event rebuild)
        self.changed_objects.difference_update(valid_ids)
        # Remove from migrated_obj_ids
        self.migrated_obj_ids.difference_update(valid_ids)

        if self.verbose and len(valid_ids) > 0:
            print(f"[Rank {self.rank}] Marked {len(valid_ids)} objects as deleted")

    def mark_global_state_changed(self):
        """Mark the global state as changed (temperature, external field, ...)"""
        self.global_state_changed = True

        if self.verbose:
            print(f"[Rank {self.rank}] Global state marked as changed")

    def clear_changed_objects(self):
        """Clear the set of changed objects"""
        self.changed_objects.clear()

    def get_deleted_objects(self) -> Set[int]:
        """
        Get the set of object IDs deleted in the previous step

        Returns:
            set of deleted object IDs
        """
        return self.deleted_objects

    def clear_deleted_objects(self):
        """Clear the deleted object set (called once ghost events are gone)"""
        self.deleted_objects.clear()

    def reset_global_state_flag(self):
        """Reset the global state change flag"""
        self.global_state_changed = False

    def set_force_rebuild(self, zero: bool = False, single: bool = False):
        """
        Set the forced rebuild flags (for external field changes and the like)

        Args:
            zero: force a rebuild of zero-object events
            single: force a rebuild of single-object events

        Example:
            # the external field changed, so all rates must be recomputed
            manager.set_force_rebuild(zero=True, single=True)
        """
        self.force_rebuild_zero = zero
        self.force_rebuild_single = single

        if self.verbose:
            if zero:
                print(f"[Rank {self.rank}] Force rebuild zero-obj events enabled")
            if single:
                print(f"[Rank {self.rank}] Force rebuild single-obj events enabled")

    def clear_force_rebuild_flags(self):
        """Clear the forced rebuild flags"""
        self.force_rebuild_zero = False
        self.force_rebuild_single = False

    # ========================================
    # Coin persistence interface (migrated object tracking)
    # ========================================

    def add_migrated_obj_ids(self, obj_ids):
        """Add IDs of objects migrated in (called by the receiver)"""
        self.migrated_obj_ids.update(oid for oid in obj_ids if oid >= 0)

    def remove_migrated_obj_ids(self, obj_ids):
        """Remove IDs of objects migrated out (called by the sender)"""
        self.migrated_obj_ids.difference_update(obj_ids)

    def get_migrated_obj_ids(self) -> Set[int]:
        """Get the current set of migrated object IDs"""
        return self.migrated_obj_ids

    def clear_migrated_obj_ids(self):
        """Clear the migrated object ID set (called on a full rebuild)"""
        self.migrated_obj_ids.clear()

    # ========================================
    # MPI synchronization interface
    # ========================================

    def sync_changed_objects(self):
        """
        Synchronize changed objects (MPI Allgather)

        Merges the changed objects of every rank onto every rank
        """
        local_changed = list(self.changed_objects)
        all_changed = self.comm.allgather(local_changed)

        for changed_list in all_changed:
            self.changed_objects.update(changed_list)

        if self.verbose and len(self.changed_objects) > 0:
            print(f"[Rank {self.rank}] After sync: {len(self.changed_objects)} changed objects")

    def sync_global_state(self):
        """
        Synchronize the global state change flag (MPI Allreduce)

        If any rank flags a global state change, every rank flags it
        """
        local_flag = np.array([1 if self.global_state_changed else 0], dtype=np.int32)
        global_flag = np.zeros(1, dtype=np.int32)
        self.comm.Allreduce(local_flag, global_flag, op=MPI.MAX)
        self.global_state_changed = (global_flag[0] > 0)

    # ========================================
    # Rebuild decision interface
    # ========================================

    def should_rebuild_zero_obj_events(self) -> bool:
        """
        Decide whether zero-object events must be rebuilt

        Conditions:
        1. first step
        2. incremental update disabled
        3. global state changed
        4. user-forced rebuild (force_rebuild_zero)

        Returns:
            True if should rebuild, False otherwise
        """
        if self.first_step:
            return True

        if not self.enabled:
            return True

        if self.global_state_changed:
            return True

        if self.force_rebuild_zero:
            if self.verbose:
                print(f"[Rank {self.rank}] Force rebuild zero-obj events (user flag)")
            return True

        self.stats['skipped_zero_obj'] += 1
        return False

    def should_rebuild_single_obj_events(self) -> bool:
        """
        Decide whether single-object events must be rebuilt

        Conditions:
        1. first step
        2. incremental update disabled
        3. objects changed
        4. [fix Bug 2] global state changed (e.g. temperature)
        5. user-forced rebuild (force_rebuild_single)

        [fix Bug 2] The global_state_changed check was added because
        the rate of a single-object event also changes when a global physics
        parameter changes: an Arrhenius rate nu*exp(-E_m/kT) depends on T, yet
        the earlier code ignored global_state_changed and reused stale rates.
        Fix: check global_state_changed, as should_rebuild_zero_obj_events does.

        Returns:
            True if should rebuild, False otherwise
        """
        if self.first_step:
            return True

        if not self.enabled:
            return True

        if len(self.changed_objects) > 0:
            return True

        # [fix Bug 2] A global state change also forces a single-object rebuild
        # Reason: single-object rates may depend on global parameters (e.g. temperature)
        if self.global_state_changed:
            if self.verbose:
                print(f"[Rank {self.rank}] Rebuild single-obj events due to global state change")
            return True

        if self.force_rebuild_single:
            if self.verbose:
                print(f"[Rank {self.rank}] Force rebuild single-obj events (user flag)")
            return True

        self.stats['skipped_single_obj'] += 1
        return False

    def should_rebuild_pair_obj_events(self) -> bool:
        """
        Decide whether pair events must be rebuilt

        Pair events always require a full rebuild (a KDTree spatial query is needed)

        Returns:
            Always True
        """
        return True

    # ========================================
    # State management interface
    # ========================================

    def step_completed(self):
        """
        Called once a step has completed

        Clears change tracking, updates statistics, clears the forced rebuild flags
        """
        if self.first_step:
            self.first_step = False
            self.stats['full_rebuilds'] += 1
        elif not self.enabled or self.global_state_changed or len(self.changed_objects) > 0 or self.force_rebuild_zero or self.force_rebuild_single:
            self.stats['full_rebuilds'] += 1
        else:
            self.stats['incremental_updates'] += 1

        # [fix] changed_objects is not cleared here; the next _build_events uses it
        # self.clear_changed_objects()
        self.reset_global_state_flag()
        self.clear_force_rebuild_flags()  # clear the forced rebuild flags automatically

    def get_changed_objects(self) -> Set[int]:
        """Get the set of changed objects"""
        return self.changed_objects.copy()

    def set_enabled(self, enabled: bool):
        """Enable or disable incremental update"""
        self.enabled = enabled

    def reset(self):
        """Reset the manager state"""
        self.changed_objects.clear()
        self.migrated_obj_ids.clear()
        self.global_state_changed = False
        self.first_step = True
        self.stats = {
            'full_rebuilds': 0,
            'incremental_updates': 0,
            'skipped_zero_obj': 0,
            'skipped_single_obj': 0,
        }

    def get_stats(self) -> dict:
        """Get the statistics"""
        return self.stats.copy()


# ===== Factory function =====

def create_incremental_update_manager(comm: MPI.Comm,
                                      enabled: bool = True,
                                      verbose: bool = False) -> IncrementalUpdateManager:
    """
    Factory: create an incremental update manager

    Args:
        comm: MPI communicator
        enabled: whether incremental update is enabled
        verbose: whether to print details

    Returns:
        an IncrementalUpdateManager instance
    """
    return IncrementalUpdateManager(comm, enabled, verbose)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"[Rank {rank}] IncrementalUpdateManager test")

    # Create the manager
    manager = create_incremental_update_manager(comm, enabled=True, verbose=(rank == 0))

    # Test the first step
    print(f"[Rank {rank}] First step: zero={manager.should_rebuild_zero_obj_events()}, "
          f"single={manager.should_rebuild_single_obj_events()}, "
          f"pair={manager.should_rebuild_pair_obj_events()}")

    manager.step_completed()

    # Test with no changes
    print(f"[Rank {rank}] No change: zero={manager.should_rebuild_zero_obj_events()}, "
          f"single={manager.should_rebuild_single_obj_events()}, "
          f"pair={manager.should_rebuild_pair_obj_events()}")

    manager.step_completed()

    # Test with changes
    manager.mark_objects_changed([1, 2, 3])
    manager.sync_changed_objects()
    print(f"[Rank {rank}] With changes: zero={manager.should_rebuild_zero_obj_events()}, "
          f"single={manager.should_rebuild_single_obj_events()}, "
          f"pair={manager.should_rebuild_pair_obj_events()}")

    manager.step_completed()

    # Test a global state change
    manager.mark_global_state_changed()
    manager.sync_global_state()
    print(f"[Rank {rank}] Global change: zero={manager.should_rebuild_zero_obj_events()}, "
          f"single={manager.should_rebuild_single_obj_events()}, "
          f"pair={manager.should_rebuild_pair_obj_events()}")

    manager.step_completed()

    # Print the statistics
    print(f"[Rank {rank}] Stats: {manager.get_stats()}")
    print(f"[Rank {rank}] Test completed")
