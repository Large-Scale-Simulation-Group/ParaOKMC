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
 Runtime load balancing module

Core mechanism:
1. RMA window: every rank keeps a bool array marking whether it is idle
2. Work stealing: once an idle rank is spotted, the sender hands it (n-1)/c of the work
3. ACK protocol: guarantees delivery; the sender stays busy until the ACK arrives
4. Source binding: the receiver records status.Get_source() so messages cannot be confused

Communication protocol:
Step 1: sender -> receiver: the assignment (metadata: number of batches and events)
Step 2: receiver: updates its RMA window (marks itself busy)
Step 3: sender -> receiver: the event data (serialization-free)
Step 4: receiver -> sender: ACK
"""

import numpy as np
from mpi4py import MPI
from typing import List, Tuple, Optional

from .rma_event_pool import (
    RMAEventPool, EVENT_DTYPE, STATUS_SENT,
    classify_event_rows, CATEGORY_SINGLE, CATEGORY_PAIR, CATEGORY_ZERO,
)

# MPI tags
TAG_TASK_REQUEST = 8001   # task assignment request (metadata)
TAG_TASK_DATA = 8002      # task data
TAG_ACK = 8003            # acknowledgement


class RuntimeLoadBalancer:
    """
    Runtime load balancer

    Redistributes work dynamically during rate evaluation
    """

    def __init__(self, comm: MPI.Comm,
                 enabled: bool = True,
                 check_interval: int = 32,
                 event_dtype: np.dtype = None,  # [dtype fix] new parameter
                 verbose: bool = False):
        """
        Initialize the runtime load balancer

        Args:
            comm: MPI communicator
            enabled: whether balancing is enabled
            check_interval: check interval (events computed between two checks)
            event_dtype: event data type (v1.6.3 dynamic dtype)
            verbose: whether to print details
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.enabled = enabled
        self.check_interval = check_interval
        self._base_check_interval = check_interval  # baseline configured by the user
        self.verbose = verbose

        # [dtype fix] Use the dynamic dtype passed in, or the default when None
        self.event_dtype = event_dtype if event_dtype is not None else EVENT_DTYPE

        # ACK counter: number of outstanding tasks
        self.pending_acks = 0

        # [determinism] Send sequence number, broadcast in the header; the receiver
        # reorders deterministically by (source, seq) - MPI only guarantees
        self._send_seq = 0

        # [deadlock fix] List of non-blocking send requests
        # Keeping the Request objects returned by Isend avoids a Send deadlock
        self.pending_send_requests = []

        # Coin persistence: reference to the incremental_manager injected by the engine
        self.incremental_manager = None

        # RMA window: holds the idle status
        # idle_status[i] = 1 means rank i is idle, 0 means busy
        self.idle_status_array = np.ones(self.size, dtype=np.int32)
        self.idle_status_array[self.rank] = -1  # our own slot is unused

        # Create the RMA window
        itemsize = MPI.INT.Get_size()
        nbytes = self.size * itemsize
        self.idle_status_win = MPI.Win.Allocate(
            nbytes, itemsize, comm=comm
        )

        # Initialize the window contents
        buf = np.ndarray(buffer=self.idle_status_win.tomemory(),
                        dtype=np.int32, shape=(self.size,))
        buf[:] = self.idle_status_array

        # Statistics
        self.stats = {
            'total_tasks_sent': 0,
            'total_tasks_received': 0,
            'total_steals': 0,
        }

        if self.verbose and self.rank == 0:
            print(f"[RuntimeLoadBalancer] Initialized:")
            print(f"  - Enabled: {enabled}")
            print(f"  - Check interval: {check_interval}")

    def mark_idle(self):
        """
        Mark this rank as idle

        Updates the window of every other rank with an RMA Put
        """
        if not self.enabled:
            return

        # Only pending_acks == 0 allows marking ourselves idle
        if self.pending_acks > 0:
            if self.verbose:
                print(f"[Rank {self.rank}] Cannot mark idle: pending_acks={self.pending_acks}")
            return

        # [deadlock fix] Lock/Unlock instead of Fence (passive-target RMA)
        idle_flag = np.array([1], dtype=np.int32)
        for target_rank in range(self.size):
            if target_rank != self.rank:
                self.idle_status_win.Lock(target_rank, MPI.LOCK_EXCLUSIVE)
                self.idle_status_win.Put(
                    [idle_flag, MPI.INT],
                    target_rank,
                    self.rank  # target_disp
                )
                self.idle_status_win.Unlock(target_rank)

        if self.verbose:
            print(f"[Rank {self.rank}] Marked idle")

    def mark_busy(self):
        """
        Mark this rank as busy

        [deadlock fix] Lock/Unlock instead of Fence (passive-target RMA)
        """
        if not self.enabled:
            return

        # [deadlock fix] Update the window of each other rank with Lock/Unlock
        busy_flag = np.array([0], dtype=np.int32)
        for target_rank in range(self.size):
            if target_rank != self.rank:
                self.idle_status_win.Lock(target_rank, MPI.LOCK_EXCLUSIVE)
                self.idle_status_win.Put(
                    [busy_flag, MPI.INT],
                    target_rank,
                    self.rank  # target_disp
                )
                self.idle_status_win.Unlock(target_rank)

        if self.verbose:
            print(f"[Rank {self.rank}] Marked busy")

    def update_check_interval(self, total_events: int):
        """
        Adapt the check interval to the current number of events.

        Rule: max(base, total_events // 8), capped at 2048.
        - few events (<256): use the baseline (32 by default) for a fast response
        - many events (10000): interval = 1250, about 8 checks per step
        """
        if not self.enabled:
            return
        adaptive = max(self._base_check_interval, total_events // 8)
        self.check_interval = min(adaptive, 2048)

    def scan_idle_ranks(self) -> List[int]:
        """
        Scan for idle ranks

        [deadlock fix] Lock/Sync/Unlock refreshes the local memory view

        Returns:
            list of idle ranks
        """
        if not self.enabled:
            return []

        # [deadlock fix] Lock/Sync/Unlock guarantees visibility
        self.idle_status_win.Lock(self.rank, MPI.LOCK_SHARED)
        self.idle_status_win.Sync()  # refresh the local memory view
        self.idle_status_win.Unlock(self.rank)

        # Read the local window contents
        buf = np.ndarray(buffer=self.idle_status_win.tomemory(),
                        dtype=np.int32, shape=(self.size,))

        # Find every idle rank
        idle_ranks = []
        for i in range(self.size):
            if i != self.rank and buf[i] == 1:
                idle_ranks.append(i)

        return idle_ranks

    def send_tasks(self, target_rank: int, events_to_send: np.ndarray):
        """
        Send work to a target rank

        Args:
            target_rank: target rank
            events_to_send: array of events to send (EVENT_DTYPE)
        """
        if not self.enabled:
            return

        num_events = len(events_to_send)

        if self.verbose:
            print(f"[Rank {self.rank}] Sending {num_events} tasks to Rank {target_rank}")

        # [deadlock fix] Non-blocking Isend instead of a blocking Send
        # Reason: if rank A sends to B while B sends to A, both block inside Send.
        # Isend lets the sender continue and receive, breaking the circular wait

        # Step 1: send the metadata (the assignment)
        # Metadata: [num_batches, events_per_batch, send_seq]
        # Simplification: one batch holding all events
        # [determinism] send_seq lets the receiver reorder by (source, seq)
        self._send_seq += 1
        metadata = np.array([1, num_events, self._send_seq], dtype=np.int32)
        req1 = self.comm.Isend([metadata, MPI.INT], dest=target_rank, tag=TAG_TASK_REQUEST)
        self.pending_send_requests.append(req1)

        if self.verbose:
            print(f"[Rank {self.rank}] Sent metadata to Rank {target_rank}: batches=1, events={num_events}")

        # Step 2: send the event data (serialization-free)
        req2 = self.comm.Isend([events_to_send, MPI.BYTE], dest=target_rank, tag=TAG_TASK_DATA)
        self.pending_send_requests.append(req2)

        if self.verbose:
            print(f"[Rank {self.rank}] Sent event data to Rank {target_rank}")

        # Step 3: wait for the ACK
        self.pending_acks += 1

        # Update the statistics
        self.stats['total_tasks_sent'] += num_events
        self.stats['total_steals'] += 1

    def check_incoming_tasks(self, event_pool: RMAEventPool) -> int:
        """
        Check for incoming task requests (non-blocking)

        Optimization: collect all incoming buffers and merge once, avoiding O(N^2) copies

        Args:
            event_pool: event pool (receives the incoming events)

        Returns:
            the number of events received
        """
        if not self.enabled:
            return 0

        total_received = 0
        # Collect all incoming event buffers (avoids repeated copies)
        # [determinism] Record (source, seq, buffer) and reorder by (source, seq)
        # before insertion: MPI does not order messages across sources, so appending
        # in arrival order would let jitter reshuffle the cumsum and the float summation
        all_incoming_buffers = []

        # Receive every pending task request
        while True:
            # Non-blocking probe for metadata
            status = MPI.Status()
            flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_TASK_REQUEST, status=status)

            if not flag:
                break  # no further task requests

            # A request is waiting: receive the metadata
            source_rank = status.Get_source()  # record the source rank

            metadata = np.empty(3, dtype=np.int32)
            self.comm.Recv([metadata, MPI.INT], source=source_rank, tag=TAG_TASK_REQUEST)

            num_batches = metadata[0]
            events_per_batch = metadata[1]
            send_seq = int(metadata[2])

            if self.verbose:
                print(f"[Rank {self.rank}] Received task request from Rank {source_rank}: "
                      f"batches={num_batches}, events={events_per_batch}")

            # [critical fix] The redundant mark_busy() call was removed
            # Reasons:
            # 1. a rank already calls mark_busy() when entering the _compute_rates loop
            # 2. mark_busy() is O(P) (Lock+Put+Unlock over every rank)
            # 3. calling it repeatedly inside the receive loop causes:
            # - a network storm: 100 task packets = 100 x P RMA operations
            # - lock contention: masses of redundant Lock/Unlock, a performance sink
            # - a logic error: the rank is busy already, no need to say so again
            # Removed: self.mark_busy()

            # Step 2: receive the event data (serialization-free)
            # [dtype fix] Allocate the buffer with the correct event_dtype
            events_buffer = np.empty(events_per_batch, dtype=self.event_dtype)
            self.comm.Recv([events_buffer, MPI.BYTE], source=source_rank, tag=TAG_TASK_DATA)

            if self.verbose:
                print(f"[Rank {self.rank}] Received {len(events_buffer)} events from Rank {source_rank}")

            # Collect the buffer instead of inserting at once (avoids repeated copies)
            all_incoming_buffers.append((source_rank, send_seq, events_buffer))

            total_received += len(events_buffer)

            # Step 3: send the ACK back to the source (non-blocking)
            # [deadlock fix] Isend instead of Send, so mutual sends cannot deadlock
            ack = np.array([1], dtype=np.int32)
            req = self.comm.Isend([ack, MPI.INT], dest=source_rank, tag=TAG_ACK)
            self.pending_send_requests.append(req)

            if self.verbose:
                print(f"[Rank {self.rank}] Sent ACK to Rank {source_rank}")

            # Update the statistics
            self.stats['total_tasks_received'] += len(events_buffer)

        # Merge all incoming events at once (avoids O(N^2) copies)
        # [determinism] Reorder by (source, seq); never trust the arrival order
        if all_incoming_buffers:
            all_incoming_buffers.sort(key=lambda item: (item[0], item[1]))
            merged_buffer = np.concatenate([b for _, _, b in all_incoming_buffers])
            self._add_events_to_pool(event_pool, merged_buffer)

            # Coin persistence: record the obj_ids of the received single-object events
            # [fix] Classify by the id_high prefix (replacing the non-negative obj_ids count)
            if self.incremental_manager is not None:
                cats = classify_event_rows(merged_buffer)
                received_obj_ids = [
                    int(event['obj_ids'][0])
                    for event, cat in zip(merged_buffer, cats)
                    if cat == CATEGORY_SINGLE
                ]
                if received_obj_ids:
                    self.incremental_manager.add_migrated_obj_ids(received_obj_ids)

        return total_received

    def check_acks(self):
        """
        Check for arriving ACKs (non-blocking)
        """
        if not self.enabled:
            return

        # Receive every pending ACK
        while self.pending_acks > 0:
            # Non-blocking probe
            status = MPI.Status()
            flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_ACK, status=status)

            if not flag:
                break  # no further ACKs

            # Receive the ACK
            source_rank = status.Get_source()
            ack = np.empty(1, dtype=np.int32)
            self.comm.Recv([ack, MPI.INT], source=source_rank, tag=TAG_ACK)

            self.pending_acks -= 1

            if self.verbose:
                print(f"[Rank {self.rank}] Received ACK from Rank {source_rank}, "
                      f"pending_acks={self.pending_acks}")

        # [deadlock fix] Reap the completed non-blocking send requests
        # Test avoids blocking while checking for completion
        self.pending_send_requests = [req for req in self.pending_send_requests if not req.Test()]

    def _add_events_to_pool(self, event_pool: RMAEventPool, events: np.ndarray):
        """
        [Bug 4 fix] Append the received events to the pool safely

        Uses append_events, so no array replacement can race.

        Args:
            event_pool: event pool
            events: array of events
        """
        # Classify the events
        # [fix] Classify by the id_high prefix (replacing the non-negative obj_ids count),
        # so a per-cell zero event with obj_ids[0]=cell_id no longer lands in single/pair
        cats = classify_event_rows(events)
        zero_events = []
        single_events = []
        pair_events = []

        for event, cat in zip(events, cats):
            if cat == CATEGORY_ZERO:
                zero_events.append(event)
            elif cat == CATEGORY_SINGLE:
                single_events.append(event)
            else:
                pair_events.append(event)

        # [Bug 4 fix] append_events is safe: no array replacement inside the compute loop
        # [dtype fix] Use the correct event_dtype
        event_pool.append_events(
            zero_events=np.array(zero_events, dtype=self.event_dtype) if zero_events else None,
            single_events=np.array(single_events, dtype=self.event_dtype) if single_events else None,
            pair_events=np.array(pair_events, dtype=self.event_dtype) if pair_events else None
        )

        if self.verbose:
            print(f"[Rank {self.rank}] Added events to pool: "
                  f"zero={len(zero_events)}, single={len(single_events)}, pair={len(pair_events)}")

    def redistribute_if_needed(self, event_pool: RMAEventPool, computed_count: int) -> bool:
        """
        Redistribute the work if necessary

        Args:
            event_pool: event pool
            computed_count: number of events computed (tells whether a check is due)

        Returns:
            whether a redistribution took place
        """
        if not self.enabled:
            return False

        # Is a check due?
        if computed_count % self.check_interval != 0:
            return False

        # First look for incoming work
        self.check_incoming_tasks(event_pool)

        # Check the ACKs
        self.check_acks()

        # Count the pending events
        remaining_events = self._count_todo_events(event_pool)

        if remaining_events == 0:
            # Nothing left to do: mark ourselves idle
            self.mark_idle()
            return False

        # Scan for idle ranks
        idle_ranks = self.scan_idle_ranks()

        # Debug: report the scan result
        if self.verbose and self.rank == 0:
            print(f"[RuntimeLB] Check at computed={computed_count}: remaining={remaining_events}, idle_ranks={idle_ranks}")

        if len(idle_ranks) == 0:
            # No idle rank
            return False

        # Number of tasks to send: (remaining - 1) / idle_count
        tasks_per_rank = max(1, (remaining_events - 1) // len(idle_ranks))

        if self.verbose:
            print(f"[Rank {self.rank}] Found {len(idle_ranks)} idle ranks, "
                  f"remaining={remaining_events}, will send {tasks_per_rank} per rank")

        # Collect the pending events
        todo_events = self._collect_todo_events(event_pool)

        # Send work to each idle rank
        sent_count = 0
        for idle_rank in idle_ranks:
            if sent_count >= len(todo_events):
                break

            # Choose the events to send
            end_idx = min(sent_count + tasks_per_rank, len(todo_events))
            events_to_send = todo_events[sent_count:end_idx]

            # Send the work
            self.send_tasks(idle_rank, events_to_send)

            sent_count += len(events_to_send)

        # Mark the sent events STATUS_SENT (deleting them would invalidate the iterator)
        if sent_count > 0:
            self._mark_events_as_sent(event_pool, todo_events[:sent_count])
            if self.verbose:
                print(f"[RuntimeLB] Rank {self.rank} offloaded {sent_count} events to {len(idle_ranks)} idle ranks")

        return sent_count > 0

    def _count_todo_events(self, event_pool: RMAEventPool) -> int:
        """Count the pending events (status=0)"""
        count = 0

        if event_pool.count_zero > 0:
            zero_events = event_pool.zero_obj_array[:event_pool.count_zero]
            count += np.sum(zero_events['status'] == 0)

        if event_pool.count_single > 0:
            single_events = event_pool.single_obj_array[:event_pool.count_single]
            count += np.sum(single_events['status'] == 0)

        if event_pool.count_pair > 0:
            pair_events = event_pool.pair_obj_array[:event_pool.count_pair]
            count += np.sum(pair_events['status'] == 0)

        return count

    def _collect_todo_events(self, event_pool: RMAEventPool) -> np.ndarray:
        """Collect all pending events (status=0)"""
        todo_list = []

        if event_pool.count_zero > 0:
            zero_events = event_pool.zero_obj_array[:event_pool.count_zero]
            mask = zero_events['status'] == 0
            if np.any(mask):
                todo_list.append(zero_events[mask])

        if event_pool.count_single > 0:
            single_events = event_pool.single_obj_array[:event_pool.count_single]
            mask = single_events['status'] == 0
            if np.any(mask):
                todo_list.append(single_events[mask])

        if event_pool.count_pair > 0:
            pair_events = event_pool.pair_obj_array[:event_pool.count_pair]
            mask = pair_events['status'] == 0
            if np.any(mask):
                todo_list.append(pair_events[mask])

        if todo_list:
            return np.concatenate(todo_list)
        else:
            # [dtype fix] Use the correct event_dtype
            return np.array([], dtype=self.event_dtype)

    def _mark_events_as_sent(self, event_pool: RMAEventPool, events_to_send: np.ndarray):
        """
        Mark the sent events STATUS_SENT instead of deleting them

        This avoids invalidating the iterator of the _compute_rates loop.
        Events marked SENT are skipped and their rate is set to 0.0 (excluded from the total).

        [fix] Events are matched by their 128-bit ID (id_high, id_low)
        The old code matched on the id field, but in 128-bit mode every event has id 0,
        so every event was wrongly marked SENT.

        Args:
            event_pool: event pool
            events_to_send: array of events to mark
        """
        # [fix] The (id_high, id_low) tuple is the unique key
        use_128bit = ('id_high' in events_to_send.dtype.names and
                      'id_low' in events_to_send.dtype.names)

        if use_128bit:
            ids_to_mark = set(
                (int(e['id_high']), int(e['id_low'])) for e in events_to_send
            )
        else:
            ids_to_mark = set(int(e['id']) for e in events_to_send)

        def _mark_array(events_array, count):
            if count == 0:
                return
            for i in range(count):
                if use_128bit:
                    key = (int(events_array[i]['id_high']),
                           int(events_array[i]['id_low']))
                else:
                    key = int(events_array[i]['id'])
                if key in ids_to_mark:
                    events_array[i]['status'] = STATUS_SENT
                    events_array[i]['rate'] = 0.0

        _mark_array(event_pool.zero_obj_array, event_pool.count_zero)
        _mark_array(event_pool.single_obj_array, event_pool.count_single)
        _mark_array(event_pool.pair_obj_array, event_pool.count_pair)

    def get_stats(self):
        """Get the statistics"""
        return self.stats.copy()

    def destroy(self):
        """Destroy the load balancer"""
        if self.enabled:
            self.idle_status_win.Free()

        if self.verbose and self.rank == 0:
            print("[RuntimeLoadBalancer] Destroyed")


def create_runtime_load_balancer(comm: MPI.Comm,
                                 enabled: bool = True,
                                 check_interval: int = 32,
                                 event_dtype: np.dtype = None,  # [dtype fix] new parameter
                                 verbose: bool = False) -> RuntimeLoadBalancer:
    """
    Factory: create a runtime load balancer

    Args:
        comm: MPI communicator
        enabled: whether balancing is enabled
        check_interval: check interval
        event_dtype: event data type (v1.6.3 dynamic dtype)
        verbose: whether to print details

    Returns:
        a RuntimeLoadBalancer instance
    """
    return RuntimeLoadBalancer(comm, enabled, check_interval, event_dtype, verbose)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print(f"[Rank {rank}] RuntimeLoadBalancer module loaded successfully")
