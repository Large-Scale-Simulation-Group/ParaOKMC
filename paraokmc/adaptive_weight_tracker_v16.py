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
Adaptive weight tracker

Features:
1. Record the actual computation time of each event type (rate evaluation stage)
2. Refresh the weight estimates periodically from those measured times
3. Apply the updated weights to the predictive load balancer automatically

How it works:
- _compute_rates() records the rate evaluation time of every event
- update_weights() is called every update_interval steps
- MPI.allreduce collects global statistics and relative weights are computed
- config['event_weights'] is updated for the next predictive balancing round
"""

import time
import numpy as np
from mpi4py import MPI
from typing import Dict, Optional
from collections import defaultdict


class AdaptiveWeightTracker:
    """Adaptive weight tracker"""

    def __init__(self,
                 comm: MPI.Comm,
                 update_interval: int = 1000,
                 base_event_type: Optional[str] = None,
                 enabled: bool = True,
                 verbose: bool = False):
        """
        Initialize the weight tracker

        Args:
            comm: MPI communicator
            update_interval: weight update interval (steps)
            base_event_type: reference event type (weight = 1.0); None picks one automatically
            enabled: whether tracking is enabled
            verbose: whether to print details
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.update_interval = update_interval
        self.base_event_type = base_event_type
        self.enabled = enabled
        self.verbose = verbose

        # Local statistics: event_type -> {'total_time': float, 'total_count': int}
        self.local_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: {'total_time': 0.0, 'total_count': 0})

        # Global averages: event_type -> {'avg_time': float, 'weight': float}
        self.global_stats: Dict[str, Dict[str, float]] = {}

        # Weight map currently in use
        self.current_weights: Dict[str, float] = {}

        # Step of the last update
        self.last_update_step = 0

        if self.verbose and self.rank == 0:
            print(f"[AdaptiveWeightTracker] Initialized")
            print(f"  - Update interval: {update_interval} steps")
            print(f"  - Base event type: {base_event_type if base_event_type else 'auto'}")
            print(f"  - Enabled: {enabled}")

    def record_event_computation(self, event_type: str, computation_time: float):
        """
        Record the rate evaluation time of an event

        Args:
            event_type: event type (string)
            computation_time: rate evaluation time (seconds)
        """
        if not self.enabled:
            return

        stats = self.local_stats[event_type]
        stats['total_time'] += computation_time
        stats['total_count'] += 1

    def record_batch_computation(self, event_type: str, count: int, total_time: float):
        """
        [fix] Record event computation times in bulk

        Used by the vectorized path to record a whole batch at once

        Args:
            event_type: event type (string)
            count: number of events
            total_time: total computation time of the batch (seconds)

        Note:
            - accumulating totals directly keeps the statistics exact
            - avoids the Python loop overhead of recording one by one
        """
        if not self.enabled or count <= 0:
            return

        stats = self.local_stats[event_type]
        stats['total_time'] += total_time
        stats['total_count'] += count

    def should_update_weights(self, current_step: int) -> bool:
        """Decide whether the weights should be updated"""
        if not self.enabled:
            return False

        return (current_step - self.last_update_step) >= self.update_interval

    def update_weights(self, current_step: int) -> Dict[str, float]:
        """
        Update the weight estimates (MPI allreduce)

        Args:
            current_step: current step

        Returns:
            the updated weight map {event_type: weight}
        """
        if not self.enabled:
            return self.current_weights

        self.last_update_step = current_step

        # Step 1: collect all event types
        local_types = set(self.local_stats.keys())
        all_types_list = self.comm.allgather(list(local_types))
        all_event_types = set()
        for types in all_types_list:
            all_event_types.update(types)

        if len(all_event_types) == 0:
            if self.verbose and self.rank == 0:
                print(f"[AdaptiveWeightTracker] Step {current_step}: No events to track yet")
            return self.current_weights

        # Step 2: allreduce for every event type
        global_stats = {}
        for event_type in all_event_types:
            if event_type in self.local_stats:
                local_time = self.local_stats[event_type]['total_time']
                local_count = self.local_stats[event_type]['total_count']
            else:
                local_time = 0.0
                local_count = 0

            global_time = self.comm.allreduce(local_time, op=MPI.SUM)
            global_count = self.comm.allreduce(local_count, op=MPI.SUM)

            if global_count > 0:
                avg_time = global_time / global_count
                global_stats[event_type] = {
                    'total_time': global_time,
                    'total_count': global_count,
                    'avg_time': avg_time
                }

        self.global_stats = global_stats

        # Step 3: choose the reference event type (if not given)
        if self.base_event_type is None and global_stats:
            self.base_event_type = sorted(global_stats.keys())[0]

        # Step 4: compute relative weights
        if self.base_event_type and self.base_event_type in global_stats:
            base_time = global_stats[self.base_event_type]['avg_time']

            if base_time > 1e-12:  # avoid division by zero
                for event_type, stats in global_stats.items():
                    weight = stats['avg_time'] / base_time
                    stats['weight'] = weight
                    self.current_weights[event_type] = weight
            else:
                # The reference time is too small, fall back to default weights
                for event_type in global_stats.keys():
                    self.current_weights[event_type] = 1.0
                    global_stats[event_type]['weight'] = 1.0

        # Step 5: report the update (rank 0)
        if self.verbose and self.rank == 0:
            print(f"\n{'='*70}")
            print(f"[AdaptiveWeightTracker] Step {current_step} - Weight Update")
            print(f"{'='*70}")
            print(f"Base event type: {self.base_event_type}")
            print(f"{'Event Type':<20} {'Avg Time (μs)':<15} {'Count':<10} {'Weight':<10}")
            print("-" * 70)
            for event_type in sorted(self.current_weights.keys()):
                if event_type in global_stats:
                    stats = global_stats[event_type]
                    weight = self.current_weights[event_type]
                    avg_time_us = stats['avg_time'] * 1e6  # convert to microseconds
                    print(f"{event_type:<20} {avg_time_us:<15.3f} {stats['total_count']:<10} {weight:<10.2f}")
            print(f"{'='*70}\n")

        return self.current_weights

    def get_event_weight(self, event_type: str, default: float = 1.0) -> float:
        """
        Get the weight of an event

        Args:
            event_type: event type
            default: default weight

        Returns:
            the event weight
        """
        return self.current_weights.get(event_type, default)

    def get_all_weights(self) -> Dict[str, float]:
        """Get all event weights"""
        return self.current_weights.copy()

    def get_summary_stats(self) -> Dict:
        """Get the summary statistics"""
        return {
            'enabled': self.enabled,
            'update_interval': self.update_interval,
            'last_update_step': self.last_update_step,
            'num_event_types': len(self.global_stats),
            'base_event_type': self.base_event_type,
            'current_weights': self.current_weights.copy()
        }

    def print_summary(self):
        """Print the summary (rank 0 only)"""
        if self.rank != 0 or not self.enabled:
            return

        print("\n" + "=" * 70)
        print("Adaptive weight tracking summary")
        print("=" * 70)
        print(f"Update interval:        {self.update_interval} steps")
        print(f"Last update step:       {self.last_update_step}")
        print(f"Number of event types:  {len(self.global_stats)}")
        print(f"Reference event type:   {self.base_event_type}")

        if self.global_stats:
            print(f"\nFinal weight distribution:")
            print(f"{'Event Type':<20} {'Weight':<10} {'Total Count':<12}")
            print("-" * 45)
            for event_type in sorted(self.current_weights.keys()):
                weight = self.current_weights[event_type]
                if event_type in self.global_stats:
                    count = self.global_stats[event_type]['total_count']
                else:
                    count = 0
                print(f"{event_type:<20} {weight:<10.2f} {count:<12}")

        print("=" * 70)


# ===== Factory function =====

def create_adaptive_weight_tracker(comm: MPI.Comm,
                                     update_interval: int = 1000,
                                     base_event_type: Optional[str] = None,
                                     enabled: bool = True,
                                     verbose: bool = False) -> AdaptiveWeightTracker:
    """
    Factory: create an adaptive weight tracker

    Args:
        comm: MPI communicator
        update_interval: weight update interval (steps)
        base_event_type: reference event type
        enabled: whether tracking is enabled
        verbose: whether to print details

    Returns:
        an AdaptiveWeightTracker instance
    """
    return AdaptiveWeightTracker(comm, update_interval, base_event_type, enabled, verbose)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"[Rank {rank}] AdaptiveWeightTracker module loaded successfully")

    if size >= 2:
        tracker = create_adaptive_weight_tracker(
            comm,
            update_interval=10,
            enabled=True,
            verbose=(rank == 0)
        )

        # Simulate recording data
        for step in range(20):
            # Simulate the computation time of different event types
            if rank == 0:
                tracker.record_event_computation('migration', 0.001)
                tracker.record_event_computation('capture', 0.002)
            else:
                tracker.record_event_computation('migration', 0.0015)

            # Update the weights periodically
            if tracker.should_update_weights(step):
                weights = tracker.update_weights(step)
                if rank == 0:
                    print(f"Step {step}: Updated weights = {weights}")

        # Print the summary
        tracker.print_summary()

        print(f"[Rank {rank}] Test completed")
    else:
        print(f"[Rank {rank}] Test requires at least 2 MPI ranks")
