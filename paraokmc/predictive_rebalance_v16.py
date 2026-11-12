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
Predictive load balancing module (zero-serialization version)

Core features:
1. Predict the load of every rank from the event weights
2. Compute a redistribution plan with a global greedy algorithm
3. Iterate until the load is balanced or the iteration limit is reached
4. **Zero-serialization communication**: only weights and event IDs travel; the event data moves once, by Alltoallv

Zero-serialization optimization:
Step 1 (Gather): each rank sends only the weight array (id, float64), not the events
Step 2 (Compute): rank 0 builds the plan from the weights; the plan holds only event IDs
Step 3 (Scatter): rank 0 broadcasts the instructions (arrays of event IDs), not the event data
Step 4 (Exchange): each rank pulls the events from its local pool by ID and exchanges them with Alltoallv

Benefits:
- removes the pickle serialization overhead
- cuts the traffic of steps 1 and 3 (only (id, weight) and ID lists)
- keeps step 4 serialization-free (NumPy structured arrays travel directly)
"""

import numpy as np
from mpi4py import MPI
from typing import Dict, List, Tuple

from .rma_event_pool import (
    RMAEventPool, EVENT_DTYPE,
    classify_event_rows, classify_event_row,
    CATEGORY_SINGLE, CATEGORY_PAIR, CATEGORY_ZERO,
)



class _TempIdMap:
    """Array-backed replacement for the per-event ``{temp_id: (id_high, id_low)}`` dict.

    Temporary ids are contiguous 1..N in the order the pools were scanned, so the
    map is just two int64 arrays; ``k in m`` and ``m[k]`` keep the dict semantics the
    redistribution code relies on, without any per-event Python work at gather time.
    """

    __slots__ = ("_high", "_low", "_chunks_high", "_chunks_low", "_n")

    def __init__(self):
        self._chunks_high = []
        self._chunks_low = []
        self._high = None
        self._low = None
        self._n = 0

    def append(self, id_high: np.ndarray, id_low: np.ndarray) -> None:
        self._chunks_high.append(np.asarray(id_high, dtype=np.int64))
        self._chunks_low.append(np.asarray(id_low, dtype=np.int64))
        self._high = self._low = None
        self._n += len(id_high)

    def _finalize(self) -> None:
        if self._high is None:
            self._high = np.concatenate(self._chunks_high) if self._chunks_high else np.empty(0, dtype=np.int64)
            self._low = np.concatenate(self._chunks_low) if self._chunks_low else np.empty(0, dtype=np.int64)

    def __len__(self) -> int:
        return self._n

    def __contains__(self, key) -> bool:
        if isinstance(key, (bool, np.bool_)) or not isinstance(key, (int, np.integer)):
            return False
        return 1 <= int(key) <= self._n

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self._finalize()
        i = int(key) - 1
        return (int(self._high[i]), int(self._low[i]))


class PredictiveRebalancer:
    """
    Predictive load balancer (full version)

    Redistributes events by weight after event construction and before rate evaluation
    """

    def __init__(self, comm: MPI.Comm,
                 enabled: bool = True,
                 threshold: float = 0.2,
                 max_iterations: int = 10,
                 event_dtype: np.dtype = None,
                 verbose: bool = False,
                 check_interval: int = 100):
        """
        Initialize the predictive load balancer

        Args:
            comm: MPI communicator
            enabled: whether balancing is enabled
            threshold: load imbalance threshold (balancing runs only above it)
            max_iterations: iteration limit of the greedy algorithm
            event_dtype: event data type (added in v1.6.3, optional)
            verbose: whether to print details
            check_interval: steps between two imbalance checks; a forced call
                (first step, after a repartition) always checks
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.enabled = enabled
        self.threshold = threshold
        self.check_interval = max(1, int(check_interval))
        self.max_iterations = max_iterations if max_iterations is not None else 10  # defensive: never None
        self.verbose = verbose

        # Local ID mapping for 128-bit IDs (temp_id -> real_id_tuple)
        self.local_id_map = _TempIdMap()

        # Dynamic dtype support
        self.event_dtype = event_dtype if event_dtype is not None else EVENT_DTYPE

        # Statistics
        self.stats = {
            'total_rebalances': 0,
            'total_events_moved': 0,
            'avg_imbalance_before': 0.0,
            'avg_imbalance_after': 0.0,
            'avg_iterations': 0.0,
        }

        if self.verbose and self.rank == 0:
            print(f"[PredictiveRebalancer] Initialized (Full Algorithm):")
            print(f"  - Enabled: {enabled}")
            print(f"  - Threshold: {threshold}")
            print(f"  - Max iterations: {max_iterations}")

    def rebalance(self, event_pool: RMAEventPool,
                  incremental: bool = False, step: int = 0, force: bool = False) -> int:
        """
        Run the predictive load balancing

        Args:
            event_pool: event pool
            incremental: True = incremental mode (no pool reset), False = full mode
            step: current step; the imbalance is checked every check_interval steps
            force: check regardless of the interval (first step, after a repartition)

        Returns:
            (total events moved, sent_obj_ids, received_obj_ids)
        """
        if not self.enabled:
            return 0, set(), set()
        # A single rank has nobody to trade events with: skip the whole gather/plan/bcast
        if self.size == 1:
            return 0, set(), set()
        # Every rank evaluates the same condition, so the collectives below stay matched
        if not force and step % self.check_interval != 0:
            return 0, set(), set()

        import time
        t_start = time.time()

        # Stage 1 (every step, cheap): only the per-rank total weight and count travel.
        # The imbalance the planner would compute from the full arrays depends on these
        # totals alone, so a balanced step costs one small allgather instead of shipping
        # every event weight to rank 0 and broadcasting a plan that moves nothing.
        local_weight, local_count = self._local_load_summary(event_pool)
        all_loads = self.comm.allgather((local_weight, local_count))
        imbalance_quick = self._compute_imbalance([w for w, _ in all_loads])
        if imbalance_quick < self.threshold:
            return 0, set(), set()

        # Stage 2 (only above the threshold): full weights, plan, redistribution
        # Step 1: collect the weights and event information of every rank
        local_info = self._gather_local_info(event_pool)
        all_info = self.comm.gather(local_info, root=0)

        # Step 2: rank 0 computes the redistribution plan
        if self.rank == 0:
            redistribution_plan, imbalance_before, imbalance_after, iterations = \
                self._compute_redistribution_plan_global(all_info)
        else:
            redistribution_plan = None
            imbalance_before = 0.0
            imbalance_after = 0.0
            iterations = 0

        # Broadcast the imbalance figures (for logging)
        imbalance_before, imbalance_after, iterations = self.comm.bcast(
            (imbalance_before, imbalance_after, iterations), root=0
        )

        # Skip if the load is already balanced
        if imbalance_before < self.threshold:
            return 0, set(), set()

        # Step 3: broadcast the redistribution plan
        redistribution_plan = self.comm.bcast(redistribution_plan, root=0)

        # Step 4: execute the redistribution
        if incremental:
            events_moved, sent_obj_ids, received_obj_ids = self._execute_redistribution_incremental(
                event_pool, redistribution_plan, local_info['done_count'])
        else:
            events_moved, sent_obj_ids, received_obj_ids = self._execute_redistribution_alltoallv(
                event_pool, redistribution_plan, local_info['done_count'])

        # Measure and report the elapsed time
        elapsed = time.time() - t_start
        max_elapsed = self.comm.allreduce(elapsed, op=MPI.MAX)
        # if self.rank == 0:
        # print(f"[Predictive LB] imbalance {imbalance_before:.2%}->{imbalance_after:.2%}, local events={local_info['count']}, CPU time={max_elapsed:.3f}s")

        # Step 5: update the statistics
        self.stats['total_rebalances'] += 1
        self.stats['total_events_moved'] += events_moved
        self.stats['avg_imbalance_before'] += imbalance_before
        self.stats['avg_imbalance_after'] += imbalance_after
        self.stats['avg_iterations'] += iterations

        return events_moved, sent_obj_ids, received_obj_ids

    def _local_load_summary(self, event_pool: RMAEventPool) -> Tuple[float, int]:
        """Per-rank total weight and count of pending (status=0) events, with the same
        zero-weight fallback as _gather_local_info (weight 1.0 per event), but without
        building the weight arrays or the temp-id map."""
        total = 0.0
        count = 0
        for arr, n in ((event_pool.zero_obj_array, event_pool.count_zero),
                       (event_pool.single_obj_array, event_pool.count_single),
                       (event_pool.pair_obj_array, event_pool.count_pair)):
            if n <= 0:
                continue
            ev = arr[:n]
            mask = ev['status'] == 0
            k = int(np.count_nonzero(mask))
            if k == 0:
                continue
            count += k
            total += float(np.sum(ev['weight'][mask]))
        if total == 0.0 and count > 0:
            total = float(count)
        return total, count

    def _gather_local_info(self, event_pool: RMAEventPool) -> Dict:
        """
        Collect the local information (zero-serialization version)

        New strategy: collect only the weights (id, weight); the events stay put

        Returns:
            {
                'weight': total weight,
                'count': total number of events,
                'weight_array': np.array([(id, weight), ...], dtype=[('id', 'i8'), ('weight', 'f8')]),
                'done_count': number of completed events
            }
        """
        # Reset local ID map
        self.local_id_map = _TempIdMap()
        _temp_counter = 1  # monotonically increasing, avoids XOR collisions

        # Collect all pending events (status=0), one vectorised pass per pool
        todo_weights_list = []
        done_count = 0
        weight_dtype = [('id', 'i8'), ('weight', 'f8')]

        pools = (
            (event_pool.zero_obj_array, event_pool.count_zero),
            (event_pool.single_obj_array, event_pool.count_single),
            (event_pool.pair_obj_array, event_pool.count_pair),
        )
        for arr, count in pools:
            if count <= 0:
                continue
            events = arr[:count]
            mask_todo = events['status'] == 0
            n_todo = int(np.count_nonzero(mask_todo))
            done_count += count - n_todo
            if n_todo == 0:
                continue
            todo_events = events[mask_todo]
            weights = np.empty(n_todo, dtype=weight_dtype)
            names = events.dtype.names
            if 'id_high' in names and 'id_low' in names:
                # [fix] 128-bit IDs: hand out contiguous temp ids and remember the
                # (id_high, id_low) pairs as arrays instead of a per-event dict
                weights['id'] = np.arange(_temp_counter, _temp_counter + n_todo, dtype=np.int64)
                _temp_counter += n_todo
                self.local_id_map.append(todo_events['id_high'], todo_events['id_low'])
            else:
                weights['id'] = todo_events['id']
            weights['weight'] = todo_events['weight']
            todo_weights_list.append(weights)

        # Merge the weight arrays
        if todo_weights_list:
            all_weights = np.concatenate(todo_weights_list)
        else:
            all_weights = np.array([], dtype=[('id', 'i8'), ('weight', 'f8')])

        total_weight = np.sum(all_weights['weight']) if len(all_weights) > 0 else 0.0

        # When every weight is 0 (e.g. incremental update off, events rebuilt each step
        # before the rates exist), fall back to 1.0 so the rebalancer can balance by count
        if total_weight == 0.0 and len(all_weights) > 0:
            all_weights['weight'] = 1.0
            total_weight = float(len(all_weights))

        return {
            'weight': float(total_weight),
            'count': len(all_weights),
            'weight_array': all_weights,  # holds only (id, weight)
            'done_count': done_count
        }

    def _compute_imbalance(self, weights: List[float]) -> float:
        """
        Compute the load imbalance

        Definition: (max_weight - avg_weight) / avg_weight
        """
        if len(weights) == 0 or sum(weights) == 0:
            return 0.0

        avg = sum(weights) / len(weights)
        max_weight = max(weights)

        if avg == 0:
            return 0.0

        return (max_weight - avg) / avg

    def _compute_redistribution_plan_global(self, all_info: List[Dict]) -> Tuple[Dict, float, float, int]:
        """
        Global greedy redistribution plan (zero-serialization version)

        New strategy: return event IDs only, saying which events move between ranks

        Algorithm:
        1. collect the weights of all ranks
        2. iterate: find the heaviest and the lightest rank
        3. move some events (by ID) from the heaviest to the lightest rank
        4. update the weights and repeat until balanced or the iteration limit is reached

        Returns:
            (redistribution_plan, imbalance_before, imbalance_after, iterations)
            redistribution_plan = {src_rank: {dst_rank: [event_ids]}}  # IDs only!
        """
        # Extract the initial weights
        initial_weights = [info['weight'] for info in all_info]
        imbalance_before = self._compute_imbalance(initial_weights)

        # Build the virtual load table (used by the greedy iteration)
        current_weights = initial_weights.copy()

        # Record the event IDs each rank should send to every other rank
        # transfer_plan[src][dst] = [event_ids]
        transfer_plan = {src: {dst: [] for dst in range(self.size)} for src in range(self.size)}

        # Keep an available-event list per rank (indices sorted by weight)
        available_events = {}
        for rank_id, info in enumerate(all_info):
            weight_array = info['weight_array']
            # Sort by descending weight (move the heavy events first)
            if len(weight_array) > 0:
                sorted_indices = np.argsort(-weight_array['weight'])  # descending
                available_events[rank_id] = {
                    'indices': sorted_indices,  # kept as a numpy array
                    'cursor': 0,  # cursor pointing at the next available event
                    'weights': weight_array  # (id, weight) array
                }
            else:
                available_events[rank_id] = {
                    'indices': np.array([], dtype=np.int64),
                    'cursor': 0,
                    'weights': np.array([], dtype=[('id', 'i8'), ('weight', 'f8')])
                }

        # Greedy iteration: sort all ranks by weight and transfer in pairs, in batches
        iterations = 0
        for iter in range(self.max_iterations):
            iterations += 1

            imbalance = self._compute_imbalance(current_weights)
            if imbalance < self.threshold:
                break

            # Sort by weight, pairing the heaviest with the lightest
            sorted_ranks = sorted(range(self.size), key=lambda r: current_weights[r])
            n_pairs = self.size // 2
            moved_any = False

            for p in range(n_pairs):
                light_rank = sorted_ranks[p]
                heavy_rank = sorted_ranks[self.size - 1 - p]
                if heavy_rank == light_rank:
                    continue

                excess = (current_weights[heavy_rank] - current_weights[light_rank]) / 2.0
                if excess <= 0:
                    continue

                avail_data = available_events[heavy_rank]
                cursor = avail_data['cursor']
                indices = avail_data['indices']

                if cursor >= len(indices):
                    continue

                # Move events in bulk from the heavy rank to the light one until excess is covered
                moved_weight = 0.0
                batch_ids = []
                while cursor < len(indices) and moved_weight < excess:
                    local_idx = indices[cursor]
                    w = avail_data['weights'][local_idx]['weight']
                    eid = avail_data['weights'][local_idx]['id']
                    batch_ids.append(int(eid))
                    moved_weight += w
                    cursor += 1

                # Update the cursor
                avail_data['cursor'] = cursor

                if batch_ids:
                    transfer_plan[heavy_rank][light_rank].extend(batch_ids)
                    current_weights[heavy_rank] -= moved_weight
                    current_weights[light_rank] += moved_weight
                    moved_any = True

            if not moved_any:
                break

        imbalance_after = self._compute_imbalance(current_weights)

        # Convert to the communication format: each rank must know what it sends and receives
        redistribution_plan = {}
        for src in range(self.size):
            send_plan = {}
            for dst in range(self.size):
                if len(transfer_plan[src][dst]) > 0:
                    # Store the event ID list only (an int64 array)
                    event_ids = np.array(transfer_plan[src][dst], dtype=np.int64)
                    send_plan[dst] = event_ids
            redistribution_plan[src] = send_plan

        return redistribution_plan, imbalance_before, imbalance_after, iterations

    def _execute_redistribution_alltoallv(self, event_pool: RMAEventPool,
                                          redistribution_plan: Dict,
                                          done_count: int) -> int:
        """
        Execute the redistribution (zero-serialization version)

        New strategy: pull the event data out of the local pool by the IDs in redistribution_plan, then exchange with Alltoallv

        Args:
            event_pool: event pool
            redistribution_plan: {src_rank: {dst_rank: event_ids_array}}  # IDs only!
            done_count: number of completed events (for logging)

        Returns:
            the number of events moved locally
        """
        if self.verbose:
            print(f"[Rank {self.rank}] Redistribution: Preserving {done_count} done events")

        # Step 1: build the local event ID -> event data mapping
        event_id_to_data = {}

        if event_pool.count_zero > 0:
            zero_events = event_pool.zero_obj_array[:event_pool.count_zero]
            for event in zero_events:
                # [fix] Every event uses the 128-bit ID
                if 'id_high' in event.dtype.names and 'id_low' in event.dtype.names:
                    event_id = (int(event['id_high']), int(event['id_low']))
                else:
                    # Backward compatible: fall back to the old id field
                    event_id = int(event['id'])
                # Defensive check: the ID must be unique
                assert event_id not in event_id_to_data, \
                    f"[Rank {self.rank}] Duplicate event ID {event_id} in zero_obj_array!"
                event_id_to_data[event_id] = event.copy()

        if event_pool.count_single > 0:
            single_events = event_pool.single_obj_array[:event_pool.count_single]
            for event in single_events:
                # [fix] Every event uses the 128-bit ID
                if 'id_high' in event.dtype.names and 'id_low' in event.dtype.names:
                    event_id = (int(event['id_high']), int(event['id_low']))
                else:
                    # Backward compatible: fall back to the old id field
                    event_id = int(event['id'])
                # Defensive check: the ID must be unique
                assert event_id not in event_id_to_data, \
                    f"[Rank {self.rank}] Duplicate event ID {event_id} in single_obj_array!"
                event_id_to_data[event_id] = event.copy()

        if event_pool.count_pair > 0:
            pair_events = event_pool.pair_obj_array[:event_pool.count_pair]
            for event in pair_events:
                # [fix] Pair events are identified by (id_high, id_low),
                # like zero/single events, so obj_ids = -1 cannot create duplicates
                if 'id_high' in event.dtype.names and 'id_low' in event.dtype.names:
                    event_id = (int(event['id_high']), int(event['id_low']))
                else:
                    # Backward compatible: fall back to the old id field
                    event_id = int(event['id'])

                # Defensive check: the ID must be unique
                assert event_id not in event_id_to_data, \
                    f"[Rank {self.rank}] Duplicate event ID {event_id} in pair_obj_array!"
                event_id_to_data[event_id] = event.copy()

        # Step 2: extract the event data to be sent, following redistribution_plan
        my_send_plan = redistribution_plan.get(self.rank, {})

        # [fix] Record the real IDs of the events to send ((id_high, id_low) tuples)
        # The old code filled sent_event_ids with temp_id (an int) while the keys of
        # event_id_to_data are tuples, so the exclusion check never matched: sent events
        sent_real_ids = set()

        if self.verbose:
            total_to_send = sum(len(ids) for ids in my_send_plan.values())
            print(f"[Rank {self.rank}] Will send {total_to_send} events")

        # Step 3: build the send buffer (in rank order, data pulled by event ID)
        send_buffers = []
        send_counts = []

        for dst_rank in range(self.size):
            if dst_rank in my_send_plan and len(my_send_plan[dst_rank]) > 0:
                event_ids = my_send_plan[dst_rank]
                # Pull the event data by ID
                events_to_send = []
                for event_id in event_ids:
                    # Resolve real ID (might be tuple for 128-bit IDs)
                    real_id = event_id
                    if event_id in self.local_id_map:
                        real_id = self.local_id_map[event_id]

                    if real_id in event_id_to_data:
                        events_to_send.append(event_id_to_data[real_id])
                        sent_real_ids.add(real_id)  # [fix] record the real ID
                    else:
                        raise RuntimeError(
                            f"[Rank {self.rank}] FATAL: Event ID {event_id} (Real: {real_id}) not found in local pool!"
                        )

                if events_to_send:
                    events_array = np.array(events_to_send, dtype=self.event_dtype)
                    send_buffers.append(events_array)
                    send_counts.append(len(events_array))
                else:
                    send_counts.append(0)
            else:
                send_counts.append(0)

        # Merge the send buffers
        if send_buffers:
            send_buffer = np.concatenate(send_buffers)
        else:
            send_buffer = np.array([], dtype=self.event_dtype)

        if self.verbose:
            print(f"[Rank {self.rank}] Calling Alltoall to exchange counts...")

        # Exchange the counts with Alltoall
        recv_counts = np.empty(self.size, dtype=np.int32)
        self.comm.Alltoall(
            [np.array(send_counts, dtype=np.int32), MPI.INT],
            [recv_counts, MPI.INT]
        )

        if self.verbose:
            print(f"[Rank {self.rank}] Alltoall completed, recv_counts={recv_counts}")

        # Compute the displacements (in units of events)
        send_displs = np.zeros(self.size, dtype=np.int32)
        recv_displs = np.zeros(self.size, dtype=np.int32)
        for i in range(1, self.size):
            send_displs[i] = send_displs[i-1] + send_counts[i-1]
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1]

        # Prepare the receive buffer
        total_recv = int(np.sum(recv_counts))
        recv_buffer = np.empty(total_recv, dtype=self.event_dtype)

        if self.verbose:
            print(f"[Rank {self.rank}] Calling Alltoallv (send={len(send_buffer)}, recv={total_recv})...")

        # [fix] MPI.BYTE needs counts/displs in bytes
        # The old code passed event counts, sending N bytes instead of N*itemsize (truncation)
        itemsize = self.event_dtype.itemsize
        send_counts_bytes = np.array(send_counts, dtype=np.int32) * itemsize
        recv_counts_bytes = recv_counts * itemsize
        send_displs_bytes = send_displs * itemsize
        recv_displs_bytes = recv_displs * itemsize

        # [deadlock fix] Call Alltoallv unconditionally
        # An MPI collective must be entered by every rank, even with an empty buffer
        self.comm.Alltoallv(
            [send_buffer, (send_counts_bytes, send_displs_bytes), MPI.BYTE],
            [recv_buffer, (recv_counts_bytes, recv_displs_bytes), MPI.BYTE]
        )

        if self.verbose:
            print(f"[Rank {self.rank}] Alltoallv completed")

        # Step 4: clear the local event pool
        event_pool.reset()

        if self.verbose:
            print(f"[Rank {self.rank}] Event pool reset, rebuilding...")

        # Step 5: rebuild the event pool (retained events + received events)
        final_events_list = []

        # a. Retained events (those not sent)
        for event_id, event_data in event_id_to_data.items():
            if event_id not in sent_real_ids:
                final_events_list.append(event_data)

        # b. Received events
        if len(recv_buffer) > 0:
            final_events_list.extend(recv_buffer)

        if not final_events_list:
            # If nothing is left, return right away
            if self.verbose:
                print(f"[Rank {self.rank}] No events after redistribution")
            events_sent = int(np.sum(send_counts))
            total_moved = self.comm.allreduce(events_sent, op=MPI.SUM)
            # Coin: report the sent/received obj_ids even with no events left
            sent_obj_ids = set()
            for real_id in sent_real_ids:
                event_data = event_id_to_data.get(real_id)
                if event_data is not None and classify_event_row(event_data) == CATEGORY_SINGLE:
                    sent_obj_ids.add(int(event_data['obj_ids'][0]))
            received_obj_ids = set()
            if len(recv_buffer) > 0:
                for event in recv_buffer:
                    if classify_event_row(event) == CATEGORY_SINGLE:
                        received_obj_ids.add(int(event['obj_ids'][0]))
            return total_moved, sent_obj_ids, received_obj_ids

        # Step 6: classify the events (vectorized)
        # [fix] Classify by the id_high prefix (replacing the non-negative obj_ids count)
        final_events_array = np.array(final_events_list, dtype=self.event_dtype)
        cats = classify_event_rows(final_events_array)

        mask_zero = cats == CATEGORY_ZERO
        mask_single = cats == CATEGORY_SINGLE
        mask_pair = cats == CATEGORY_PAIR

        # Step 7: call the replacement method (bulk replace)
        event_pool.set_zero_obj_events(final_events_array[mask_zero])
        event_pool.set_single_obj_events(final_events_array[mask_single])
        event_pool.set_pair_obj_events(final_events_array[mask_pair])

        # [DEBUG] Check the rebuilt single events for duplicates (verbose mode only)
        if self.verbose and np.any(mask_single):
            _fs = final_events_array[mask_single]
            _eids = list(zip(_fs['id_high'].tolist(), _fs['id_low'].tolist()))
            _n_uniq = len(set(_eids))
            if _n_uniq < len(_eids):
                print(f"[Rank {self.rank}] [DEBUG-rebal] Rebalancer output has {len(_eids)-_n_uniq} single dups! "
                      f"retained={sum(1 for eid in event_id_to_data if eid not in sent_real_ids)}, "
                      f"received={len(recv_buffer)}, total_single={int(np.sum(mask_single))}")

        if self.verbose:
            print(f"[Rank {self.rank}] Pool rebuilt: zero={int(np.sum(mask_zero))}, single={int(np.sum(mask_single))}, pair={int(np.sum(mask_pair))}")

        # Count the events moved
        events_sent = int(np.sum(send_counts))
        total_moved = self.comm.allreduce(events_sent, op=MPI.SUM)

        # Coin persistence: collect the obj_ids of sent/received single-object events
        sent_obj_ids = set()
        for real_id in sent_real_ids:
            event_data = event_id_to_data.get(real_id)
            if event_data is not None:
                if classify_event_row(event_data) == CATEGORY_SINGLE:
                    sent_obj_ids.add(int(event_data['obj_ids'][0]))

        received_obj_ids = set()
        if len(recv_buffer) > 0:
            single_mask = classify_event_rows(recv_buffer) == CATEGORY_SINGLE
            if np.any(single_mask):
                received_obj_ids = set(recv_buffer[single_mask]['obj_ids'][:, 0].tolist())

        return total_moved, sent_obj_ids, received_obj_ids

    def _execute_redistribution_incremental(self, event_pool: RMAEventPool,
                                            redistribution_plan: Dict,
                                            done_count: int):
        """
        Incremental redistribution: no pool reset; sent events are removed and received events appended.

        Differences from _execute_redistribution_alltoallv:
        - no O(N) deep copy into a dict, only a light index (event_id -> array_name, index)
        - event_pool.reset() is not called
        - sent events are filtered with a mask, received events go through append_events

        Returns:
            (total_moved, sent_obj_ids, received_obj_ids)
        """
        if self.verbose:
            print(f"[Rank {self.rank}] Incremental redistribution: Preserving {done_count} done events")

        # Step 1: build the light index event_id -> (array_name, index)
        # No deep copy; only the positions are recorded
        event_id_index = {}
        array_map = {
            'zero': (event_pool.zero_obj_array, event_pool.count_zero),
            'single': (event_pool.single_obj_array, event_pool.count_single),
            'pair': (event_pool.pair_obj_array, event_pool.count_pair),
        }

        for array_name, (events_array, count) in array_map.items():
            for i in range(count):
                event = events_array[i]
                if 'id_high' in event.dtype.names and 'id_low' in event.dtype.names:
                    event_id = (int(event['id_high']), int(event['id_low']))
                else:
                    event_id = int(event['id'])
                event_id_index[event_id] = (array_name, i)

        # Step 2: extract the events to send from the plan (copying only what is sent)
        my_send_plan = redistribution_plan.get(self.rank, {})
        sent_real_ids = set()
        send_buffers = []
        send_counts = []

        for dst_rank in range(self.size):
            if dst_rank in my_send_plan and len(my_send_plan[dst_rank]) > 0:
                event_ids = my_send_plan[dst_rank]
                events_to_send = []
                for event_id in event_ids:
                    real_id = event_id
                    if event_id in self.local_id_map:
                        real_id = self.local_id_map[event_id]

                    if real_id in event_id_index:
                        aname, idx = event_id_index[real_id]
                        arr, _ = array_map[aname]
                        events_to_send.append(arr[idx].copy())
                        sent_real_ids.add(real_id)
                    else:
                        raise RuntimeError(
                            f"[Rank {self.rank}] FATAL: Event ID {event_id} (Real: {real_id}) not found in local pool!")

                if events_to_send:
                    events_array = np.array(events_to_send, dtype=self.event_dtype)
                    send_buffers.append(events_array)
                    send_counts.append(len(events_array))
                else:
                    send_counts.append(0)
            else:
                send_counts.append(0)

        # Step 3: exchange with Alltoallv
        if send_buffers:
            send_buffer = np.concatenate(send_buffers)
        else:
            send_buffer = np.array([], dtype=self.event_dtype)

        recv_counts = np.empty(self.size, dtype=np.int32)
        self.comm.Alltoall(
            [np.array(send_counts, dtype=np.int32), MPI.INT],
            [recv_counts, MPI.INT]
        )

        send_displs = np.zeros(self.size, dtype=np.int32)
        recv_displs = np.zeros(self.size, dtype=np.int32)
        for i in range(1, self.size):
            send_displs[i] = send_displs[i-1] + send_counts[i-1]
            recv_displs[i] = recv_displs[i-1] + recv_counts[i-1]

        total_recv = int(np.sum(recv_counts))
        recv_buffer = np.empty(total_recv, dtype=self.event_dtype)

        # [fix] MPI.BYTE needs counts/displs in bytes
        itemsize = self.event_dtype.itemsize
        send_counts_bytes = np.array(send_counts, dtype=np.int32) * itemsize
        recv_counts_bytes = recv_counts * itemsize
        send_displs_bytes = send_displs * itemsize
        recv_displs_bytes = recv_displs * itemsize

        self.comm.Alltoallv(
            [send_buffer, (send_counts_bytes, send_displs_bytes), MPI.BYTE],
            [recv_buffer, (recv_counts_bytes, recv_displs_bytes), MPI.BYTE]
        )

        # [fix] Coin persistence: collect the obj_ids of the sent events before touching the pool
        # The old code read them after set_*_obj_events using stale indices, getting wrong data
        sent_obj_ids = set()
        for real_id in sent_real_ids:
            aname, idx = event_id_index[real_id]
            arr, _ = array_map[aname]
            event_data = arr[idx]
            if classify_event_row(event_data) == CATEGORY_SINGLE:
                sent_obj_ids.add(int(event_data['obj_ids'][0]))

        # Step 4: update the pool incrementally - drop the sent events (mask filter)
        if sent_real_ids:
            for array_name, (events_array, count) in array_map.items():
                if count == 0:
                    continue
                valid = events_array[:count]
                keep_mask = np.ones(count, dtype=bool)
                for i in range(count):
                    if 'id_high' in valid.dtype.names:
                        eid = (int(valid[i]['id_high']), int(valid[i]['id_low']))
                    else:
                        eid = int(valid[i]['id'])
                    if eid in sent_real_ids:
                        keep_mask[i] = False
                kept = valid[keep_mask]
                if array_name == 'zero':
                    event_pool.set_zero_obj_events(kept)
                elif array_name == 'single':
                    event_pool.set_single_obj_events(kept)
                else:
                    event_pool.set_pair_obj_events(kept)

        # Step 5: append the received events (vectorized classification)
        # [fix] Classify by the id_high prefix (replacing the non-negative obj_ids count)
        recv_cats = None
        if total_recv > 0:
            recv_cats = classify_event_rows(recv_buffer)
            r_mask_zero = recv_cats == CATEGORY_ZERO
            r_mask_single = recv_cats == CATEGORY_SINGLE
            r_mask_pair = recv_cats == CATEGORY_PAIR

            event_pool.append_events(
                zero_events=recv_buffer[r_mask_zero] if np.any(r_mask_zero) else None,
                single_events=recv_buffer[r_mask_single] if np.any(r_mask_single) else None,
                pair_events=recv_buffer[r_mask_pair] if np.any(r_mask_pair) else None,
            )

        if self.verbose:
            print(f"[Rank {self.rank}] Incremental pool update: sent={len(sent_real_ids)}, recv={total_recv}")

        # Step 6: statistics
        events_sent = int(np.sum(send_counts))
        total_moved = self.comm.allreduce(events_sent, op=MPI.SUM)

        # Coin persistence: collect the received single-object obj_ids (vectorized, reusing recv_cats)
        received_obj_ids = set()
        if total_recv > 0 and recv_cats is not None:
            single_mask = recv_cats == CATEGORY_SINGLE
            if np.any(single_mask):
                received_obj_ids = set(recv_buffer[single_mask]['obj_ids'][:, 0].tolist())

        return total_moved, sent_obj_ids, received_obj_ids

    def get_stats(self) -> Dict:
        """Get the statistics"""
        return self.stats.copy()

    def reset_stats(self):
        """Reset the statistics"""
        self.stats = {
            'total_rebalances': 0,
            'total_events_moved': 0,
            'avg_imbalance_before': 0.0,
            'avg_imbalance_after': 0.0,
            'avg_iterations': 0.0,
        }


# ===== Factory function =====

def create_predictive_rebalancer(comm: MPI.Comm,
                                  enabled: bool = True,
                                  threshold: float = 0.2,
                                  max_iterations: int = 10,
                                  event_dtype: np.dtype = None,
                                  verbose: bool = False,
                                  check_interval: int = 100) -> PredictiveRebalancer:
    """
    Factory: create a predictive load balancer

    Args:
        comm: MPI communicator
        enabled: whether balancing is enabled
        threshold: load imbalance threshold
        max_iterations: iteration limit
        event_dtype: event data type (added in v1.6.3, optional)
        verbose: whether to print details
        check_interval: steps between two imbalance checks

    Returns:
        a PredictiveRebalancer instance
    """
    return PredictiveRebalancer(comm, enabled, threshold, max_iterations, event_dtype, verbose,
                                check_interval=check_interval)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    print(f"[Rank {rank}] PredictiveRebalancer module loaded successfully")

    if size >= 2:
        rebalancer = create_predictive_rebalancer(
            comm,
            enabled=True,
            threshold=0.2,
            verbose=(rank == 0)
        )

        print(f"[Rank {rank}] Test completed")
    else:
        print(f"[Rank {rank}] Test requires at least 2 MPI ranks")
