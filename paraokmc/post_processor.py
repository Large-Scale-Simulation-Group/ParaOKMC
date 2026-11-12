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
Online post-processing module

Core features:
1. Manage several analysis callbacks
2. Run online analysis at a chosen interval
3. Support parallel analysis (every rank may take part)
4. OVITO output (LAMMPS dump format)

Typical uses:
- live statistics (particle counts, concentration profiles)
- online visualization output (OVITO)
- performance monitoring
- data export
"""

from mpi4py import MPI
from typing import Callable, List, Any, Dict, Optional, Union
import time
import numpy as np
import os
import re


def _get_object_type_name(obj, pool=None) -> str:
    if pool is not None and hasattr(pool, 'get_type_name'):
        return pool.get_type_name(obj)
    dtype = getattr(obj, 'dtype', None)
    if dtype is not None and dtype.names is not None:
        if 'type' in dtype.names:
            return str(obj['type']).strip()
        if 'type_id' in dtype.names and pool is not None:
            return getattr(pool, '_int_to_type', {}).get(int(obj['type_id']), '')
    return ''


class PostProcessor:
    """
    Online post-processing module

    Holds the analysis callbacks and runs them periodically inside the main loop
    """

    def __init__(self, comm: MPI.Comm,
                 interval: int = 100,
                 verbose: bool = False):
        """
        Initialize the post-processor

        Args:
            comm: MPI communicator
            interval: execution interval (steps)
            verbose: whether to print details
        """
        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.interval = interval
        self._last_processed_step = None   # step of the last run (guards the forced final frame)
        self.verbose = verbose

        # All registered analysis functions
        # Signature of each: func(step, time, rma_object_pool, comm) -> None
        self.callbacks: List[Callable] = []

        # Event hooks: run on every rank right after a state-change message has been
        # applied, once per KMC step. Every rank receives the same message, so a hook
        # that only reads its arguments keeps identical state on every rank without
        # any communication (an absorption counter, an event histogram, ...).
        # Signature of each:
        #   func(event_type, changed_objects, step, time, rma_object_pool, comm, config) -> None
        #   - event_type: name of the executed event class ('' if none)
        #   - changed_objects: structured array (changed_object_dtype) with the complete
        #     records of the created / modified / deleted objects; a deleted object has
        #     active=False and still carries its type, coordinates and properties
        self.event_hooks: List[Callable] = []

        # Statistics
        self.total_calls = 0
        self.total_time = 0.0
        self.total_event_calls = 0
        self.total_event_time = 0.0

        if self.verbose and self.rank == 0:
            print(f"[PostProcessor] Initialized with interval={interval}")

    def add_analysis(self, func: Callable, name: str = None):
        """
        Add an analysis function

        Args:
            func: the analysis function
                  signature: func(step, time, object_pool, comm) -> None
            name: name of the function (optional, for debugging)
        """
        self.callbacks.append(func)

        if self.verbose and self.rank == 0:
            func_name = name or func.__name__
            print(f"[PostProcessor] Added analysis function: {func_name}")
            print(f"  Total callbacks: {len(self.callbacks)}")

    def add_event_hook(self, func: Callable, name: str = None):
        """
        Add an event hook (called on every rank after each applied state-change message)

        Args:
            func: the hook
                  signature: func(event_type, changed_objects, step, time,
                                  object_pool, comm, config) -> None
            name: name of the function (optional, for debugging)
        """
        self.event_hooks.append(func)

        if self.verbose and self.rank == 0:
            func_name = name or func.__name__
            print(f"[PostProcessor] Added event hook: {func_name}")
            print(f"  Total event hooks: {len(self.event_hooks)}")

    def on_event(self, event_type: str, changed_objects: np.ndarray,
                 step: int, time: float, rma_object_pool, config: Dict[str, Any]):
        """
        Run the event hooks for the state-change message of this step

        Called by the engine on every rank, after the message has been applied to the
        object pool and before the clock advances. Cheap when no hook is registered.

        Args:
            event_type: name of the executed event class
            changed_objects: the broadcast array of changed objects (changed_object_dtype)
            step: current step (the step in which the event fired)
            time: current time (the time at which the event fired)
            rma_object_pool: RMA object pool (already updated)
            config: the engine configuration (property mappings, type ids, physics, ...)
        """
        if not self.event_hooks:
            return

        t_start = MPI.Wtime()
        for i, func in enumerate(self.event_hooks):
            try:
                func(event_type, changed_objects, step, time, rma_object_pool,
                     self.comm, config)
            except Exception as e:
                if self.rank == 0:
                    print(f"[PostProcessor] Error in event hook {i} ({func.__name__}): {e}")
        self.total_event_time += MPI.Wtime() - t_start
        self.total_event_calls += 1

    def set_interval(self, interval: int):
        """
        Change the execution interval at runtime

        Args:
            interval: the new execution interval (steps)
        """
        self.interval = interval

        if self.verbose and self.rank == 0:
            print(f"[PostProcessor] Interval updated to {interval}")

    def process(self, step: int, time: float, rma_object_pool, force: bool = False):
        """
        Run the post-processing

        Args:
            step: current step
            time: current time
            rma_object_pool: RMA object pool
        """
        # Is a run due? (force=True: run regardless of the interval, used once for the
        # final state after the step loop; a step already processed is never repeated)
        if not force and step % self.interval != 0:
            return
        if step == self._last_processed_step:
            return
        self._last_processed_step = step

        if self.verbose and self.rank == 0:
            print(f"[PostProcessor] Running analysis at step {step}")

        # Run every callback
        t_start = time if isinstance(time, float) else 0.0
        analysis_start = MPI.Wtime()

        for i, func in enumerate(self.callbacks):
            try:
                func(step, t_start, rma_object_pool, self.comm)
            except Exception as e:
                if self.rank == 0:
                    print(f"[PostProcessor] Error in callback {i} ({func.__name__}): {e}")

        analysis_elapsed = MPI.Wtime() - analysis_start
        self.total_time += analysis_elapsed
        self.total_calls += 1

        if self.verbose and self.rank == 0:
            print(f"[PostProcessor] Analysis completed in {analysis_elapsed*1000:.2f} ms")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get the statistics

        Returns:
            dict of statistics
        """
        return {
            'num_callbacks': len(self.callbacks),
            'interval': self.interval,
            'total_calls': self.total_calls,
            'total_time': self.total_time,
            'avg_time': self.total_time / self.total_calls if self.total_calls > 0 else 0.0,
            'num_event_hooks': len(self.event_hooks),
            'total_event_calls': self.total_event_calls,
            'total_event_time': self.total_event_time,
        }

    def print_summary(self):
        """Print the statistics summary (rank 0 only)"""
        if self.rank == 0:
            stats = self.get_stats()
            print("\n" + "=" * 60)
            print("PostProcessor Summary")
            print("=" * 60)
            print(f"  Number of callbacks: {stats['num_callbacks']}")
            print(f"  Execution interval: {stats['interval']} steps")
            print(f"  Total executions: {stats['total_calls']}")
            print(f"  Total time: {stats['total_time']:.4f} s")
            print(f"  Average time: {stats['avg_time']*1000:.2f} ms")
            print("=" * 60 + "\n")


# ===== Factory function =====

def create_post_processor(comm: MPI.Comm,
                          interval: int = 100,
                          verbose: bool = False) -> PostProcessor:
    """
    Factory: create a post-processor

    Args:
        comm: MPI communicator
        interval: execution interval
        verbose: whether to print details

    Returns:
        a PostProcessor instance
    """
    return PostProcessor(comm, interval, verbose)


# ===== Example built-in analysis functions =====

def analysis_object_count(step, time, pool, comm):
    """
    Built-in analysis: count the objects

    Args:
        step: current step
        time: current time
        pool: RMA object pool
        comm: MPI communicator
    """
    local_count = pool.get_local_count()
    total_count = comm.allreduce(local_count, op=MPI.SUM)

    if comm.rank == 0:
        print(f"[Analysis] Step {step:6d}, Time {time:10.4f}: Total objects = {total_count}")


def analysis_type_distribution(step, time, pool, comm):
    """
    Built-in analysis: distribution of object types

    Args:
        step: current step
        time: current time
        pool: RMA object pool
        comm: MPI communicator
    """
    local_objs = pool.get_local_objects()

    # Local type distribution
    local_types = {}
    for obj in local_objs:
        obj_type = _get_object_type_name(obj, pool)
        local_types[obj_type] = local_types.get(obj_type, 0) + 1

    # Collect the statistics of every rank
    all_types_list = comm.gather(local_types, root=0)

    if comm.rank == 0:
        # Merge the statistics
        global_types = {}
        for local_dict in all_types_list:
            for obj_type, count in local_dict.items():
                global_types[obj_type] = global_types.get(obj_type, 0) + count

        # Report
        print(f"[Analysis] Step {step}: Type distribution:")
        for obj_type, count in sorted(global_types.items()):
            print(f"  {obj_type}: {count}")


def create_csv_writer(filename: str):
    """
    Create a CSV writer (factory function)

    Returns an analysis function that writes the statistics to a CSV file

    Args:
        filename: name of the CSV file

    Returns:
        the analysis function
    """
    # Initialization flag
    header_written = [False]

    def writer_func(step, time, pool, comm):
        """CSV writing function"""
        local_count = pool.get_local_count()
        total_count = comm.allreduce(local_count, op=MPI.SUM)

        if comm.rank == 0:
            mode = 'w' if not header_written[0] else 'a'

            with open(filename, mode) as f:
                if not header_written[0]:
                    f.write("step,time,total_objects\n")
                    header_written[0] = True

                f.write(f"{step},{time:.6f},{total_count}\n")

    return writer_func


# ===== OVITO output support =====

def create_dump_writer(
    output_dir: str = "dump",
    prefix: str = "dump",
    box_size: List[float] = None,
    properties: List[str] = None,
    type_map: Dict[str, int] = None,
    periodic: List[bool] = None,
    float_props_names: Dict[str, List[str]] = None,
):
    """
    Create a LAMMPS dump writer (factory function)

    Returns an analysis function that writes the object data in LAMMPS dump format,
    which OVITO can open directly for visualization and analysis.

    Args:
        output_dir: output directory (default "dump")
        prefix: file name prefix (default "dump")
        box_size: simulation box dimensions [Lx, Ly, Lz] (required)
        properties: properties to write (default ["id", "type", "x", "y", "z"])
        type_map: mapping from type name to integer (e.g. {"H": 1, "V": 2});
                  None generates the mapping automatically
        periodic: periodic boundary conditions [px, py, pz] (default [True, True, True])
        float_props_names: names of the float properties {type: [property names]},
                          used to give the float_props array meaningful names

    Returns:
        the analysis function (can be added to a PostProcessor)

    Examples:
        # Basic use
        dump_writer = create_dump_writer(
            output_dir="output/dump",
            box_size=[100.0, 100.0, 100.0],
        )
        post_processor.add_analysis(dump_writer, name="dump_writer")

        # Advanced use: custom properties
        dump_writer = create_dump_writer(
            output_dir="output/dump",
            box_size=[100.0, 100.0, 100.0],
            properties=["id", "type", "x", "y", "z", "size", "energy"],
            type_map={"H": 1, "V": 2, "HV": 3},
            float_props_names={
                "H": ["energy", "D"],
                "V": ["energy", "D"],
            },
        )

    LAMMPS dump format:
        ITEM: TIMESTEP
        0
        ITEM: NUMBER OF ATOMS
        100
        ITEM: BOX BOUNDS pp pp pp
        0.0 100.0
        0.0 100.0
        0.0 100.0
        ITEM: ATOMS id type x y z ...
        1 1 10.0 20.0 30.0 ...
    """
    # Validate the required arguments
    if box_size is None:
        raise ValueError("box_size is required for dump writer")

    # Defaults
    if properties is None:
        properties = ["id", "type", "x", "y", "z"]
    if periodic is None:
        periodic = [True, True, True]
    if float_props_names is None:
        float_props_names = {}
    # Slot mapping {type: {prop: slot}} of user-declared object properties, installed by
    # the engine through attach_props_mapping(); a user then only lists the property name
    # under "properties". float_props_names stays available as a manual override.
    props_slots = [{}]

    # State variables (kept in lists so the closure can mutate them)
    auto_type_map = [{}]  # the automatically generated type map
    next_type_id = [1]    # the next type ID

    def get_type_id(obj_type: str) -> int:
        """Get the integer ID of a type"""
        if type_map is not None and obj_type in type_map:
            return type_map[obj_type]

        # Generate the mapping automatically
        if obj_type not in auto_type_map[0]:
            auto_type_map[0][obj_type] = next_type_id[0]
            next_type_id[0] += 1

        return auto_type_map[0][obj_type]

    def get_property_value(obj, prop_name: str, obj_type: str) -> Union[int, float, str]:
        """Get a property value of an object"""
        # Basic properties
        if prop_name == "id":
            return int(obj['id'])
        elif prop_name == "type":
            # The type NAME, as declared in config['objects']: stable across runs and rank
            # counts, imported by OVITO as a named particle type, and mapped straight back by
            # read_dump_frame(). type_map still decides the integer id written as "type_id".
            return str(obj_type)
        elif prop_name == "type_id":
            return get_type_id(str(obj_type))
        elif prop_name in ["x", "y", "z"]:
            return float(obj[prop_name])
        elif prop_name == "size":
            if 'size' in obj.dtype.names:
                return float(obj['size'])
            return 0.0
        elif prop_name == "owner_rank":
            if 'owner_rank' in obj.dtype.names:
                return int(obj['owner_rank'])
            return 0

        # Check the float_props mapping
        if obj_type in float_props_names:
            float_names = float_props_names[obj_type]
            if prop_name in float_names:
                idx = float_names.index(prop_name)
                if 'float_props' in obj.dtype.names and idx < len(obj['float_props']):
                    return float(obj['float_props'][idx])
                return 0.0
        # User-declared object property (objects[type].properties), slot from the engine
        slot = props_slots[0].get(obj_type, {}).get(prop_name)
        if slot is not None and 'float_props' in obj.dtype.names and slot < len(obj['float_props']):
            return float(obj['float_props'][slot])

        # Try to read it straight from the dtype
        if prop_name in obj.dtype.names:
            val = obj[prop_name]
            if isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            else:
                return str(val)

        return 0  # default value

    def dump_writer_func(step: int, sim_time: float, pool, comm: MPI.Comm):
        """LAMMPS dump writing function"""
        rank = comm.rank

        # Collect the local objects
        local_objs = pool.get_local_objects(active_only=True)

        # Gather every object onto rank 0
        all_objs_list = comm.gather(local_objs, root=0)

        if rank == 0:
            # Merge all objects
            if all_objs_list:
                all_objs = np.concatenate([arr for arr in all_objs_list if len(arr) > 0])
            else:
                all_objs = np.array([])

            n_atoms = len(all_objs)

            # Create the output directory
            os.makedirs(output_dir, exist_ok=True)

            # Build the file name
            filename = os.path.join(output_dir, f"{prefix}.{step}.dump")

            # Boundary condition string
            bc_str = " ".join(["pp" if p else "ff" for p in periodic])

            # Write the file
            with open(filename, 'w') as f:
                # TIMESTEP
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{step}\n")

                # NUMBER OF ATOMS
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{n_atoms}\n")

                # BOX BOUNDS
                f.write(f"ITEM: BOX BOUNDS {bc_str}\n")
                f.write(f"0.0 {box_size[0]}\n")
                f.write(f"0.0 {box_size[1]}\n")
                f.write(f"0.0 {box_size[2]}\n")

                # ATOMS header
                props_str = " ".join(properties)
                f.write(f"ITEM: ATOMS {props_str}\n")

                # Write each atom
                for obj in all_objs:
                    obj_type = _get_object_type_name(obj, pool)
                    values = [get_property_value(obj, prop, obj_type) for prop in properties]

                    # Format the output
                    formatted = []
                    for val in values:
                        if isinstance(val, int):
                            formatted.append(str(val))
                        elif isinstance(val, float):
                            formatted.append(f"{val:.6g}")
                        else:
                            formatted.append(str(val))

                    f.write(" ".join(formatted) + "\n")

    def attach_props_mapping(mapping):
        """Engine hook: install the {type: {prop: slot}} mapping of user properties."""
        props_slots[0] = dict(mapping or {})
    dump_writer_func.attach_props_mapping = attach_props_mapping
    return dump_writer_func


def create_trajectory_writer(
    output_file: str = "trajectory.dump",
    box_size: List[float] = None,
    properties: List[str] = None,
    type_map: Dict[str, int] = None,
    periodic: List[bool] = None,
    float_props_names: Dict[str, List[str]] = None,
):
    """
    Create a trajectory writer (one file, many frames)

    Like create_dump_writer, but writes every frame into a single file,
    which suits trajectory analysis of long runs.

    Args:
        output_file: path of the output file (default "trajectory.dump")
        other arguments as in create_dump_writer

    Returns:
        the analysis function (can be added to a PostProcessor)

    Example:
        traj_writer = create_trajectory_writer(
            output_file="output/trajectory.dump",
            box_size=[100.0, 100.0, 100.0],
        )
        post_processor.add_analysis(traj_writer, name="trajectory")
    """
    # Validate the required arguments
    if box_size is None:
        raise ValueError("box_size is required for trajectory writer")

    # Defaults
    if properties is None:
        properties = ["id", "type", "x", "y", "z"]
    if periodic is None:
        periodic = [True, True, True]
    if float_props_names is None:
        float_props_names = {}
    # Slot mapping {type: {prop: slot}} of user-declared object properties, installed by
    # the engine through attach_props_mapping(); a user then only lists the property name
    # under "properties". float_props_names stays available as a manual override.
    props_slots = [{}]

    # State variables
    auto_type_map = [{}]
    next_type_id = [1]
    file_initialized = [False]

    def get_type_id(obj_type: str) -> int:
        if type_map is not None and obj_type in type_map:
            return type_map[obj_type]
        if obj_type not in auto_type_map[0]:
            auto_type_map[0][obj_type] = next_type_id[0]
            next_type_id[0] += 1
        return auto_type_map[0][obj_type]

    def get_property_value(obj, prop_name: str, obj_type: str):
        if prop_name == "id":
            return int(obj['id'])
        elif prop_name == "type":
            # The type NAME, as declared in config['objects']: stable across runs and rank
            # counts, imported by OVITO as a named particle type, and mapped straight back by
            # read_dump_frame(). type_map still decides the integer id written as "type_id".
            return str(obj_type)
        elif prop_name == "type_id":
            return get_type_id(str(obj_type))
        elif prop_name in ["x", "y", "z"]:
            return float(obj[prop_name])
        elif prop_name == "size":
            if 'size' in obj.dtype.names:
                return float(obj['size'])
            return 0.0
        elif prop_name == "owner_rank":
            if 'owner_rank' in obj.dtype.names:
                return int(obj['owner_rank'])
            return 0

        if obj_type in float_props_names:
            float_names = float_props_names[obj_type]
            if prop_name in float_names:
                idx = float_names.index(prop_name)
                if 'float_props' in obj.dtype.names and idx < len(obj['float_props']):
                    return float(obj['float_props'][idx])
                return 0.0
        # User-declared object property (objects[type].properties), slot from the engine
        slot = props_slots[0].get(obj_type, {}).get(prop_name)
        if slot is not None and 'float_props' in obj.dtype.names and slot < len(obj['float_props']):
            return float(obj['float_props'][slot])

        if prop_name in obj.dtype.names:
            val = obj[prop_name]
            if isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            else:
                return str(val)

        return 0

    def trajectory_writer_func(step: int, sim_time: float, pool, comm: MPI.Comm):
        """Trajectory writing function"""
        rank = comm.rank

        local_objs = pool.get_local_objects(active_only=True)
        all_objs_list = comm.gather(local_objs, root=0)

        if rank == 0:
            if all_objs_list:
                all_objs = np.concatenate([arr for arr in all_objs_list if len(arr) > 0])
            else:
                all_objs = np.array([])

            n_atoms = len(all_objs)

            # Create the output directory
            output_dir = os.path.dirname(output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Truncate the file on the first write
            mode = 'w' if not file_initialized[0] else 'a'
            file_initialized[0] = True

            bc_str = " ".join(["pp" if p else "ff" for p in periodic])

            with open(output_file, mode) as f:
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{step}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{n_atoms}\n")
                f.write(f"ITEM: BOX BOUNDS {bc_str}\n")
                f.write(f"0.0 {box_size[0]}\n")
                f.write(f"0.0 {box_size[1]}\n")
                f.write(f"0.0 {box_size[2]}\n")

                props_str = " ".join(properties)
                f.write(f"ITEM: ATOMS {props_str}\n")

                for obj in all_objs:
                    obj_type = _get_object_type_name(obj, pool)
                    values = [get_property_value(obj, prop, obj_type) for prop in properties]
                    formatted = []
                    for val in values:
                        if isinstance(val, int):
                            formatted.append(str(val))
                        elif isinstance(val, float):
                            formatted.append(f"{val:.6g}")
                        else:
                            formatted.append(str(val))
                    f.write(" ".join(formatted) + "\n")

    def attach_props_mapping(mapping):
        """Engine hook: install the {type: {prop: slot}} mapping of user properties."""
        props_slots[0] = dict(mapping or {})
    trajectory_writer_func.attach_props_mapping = attach_props_mapping
    return trajectory_writer_func


# ===== High-performance distributed output =====

def create_distributed_dump_writer(
    output_dir: str = "dump",
    prefix: str = "dump",
    box_size: List[float] = None,
    properties: List[str] = None,
    type_map: Dict[str, int] = None,
    periodic: List[bool] = None,
    float_props_names: Dict[str, List[str]] = None,
    strict_type_check: bool = False,
):
    """
    Create a high-performance distributed dump writer

    Every rank writes its own file, avoiding the memory pressure and serial
    bottleneck of a gather. Suited to large runs (N > 10^6).

    Output file format:
        dump.{step}.rank_{rank}.dump

    Loading in OVITO:
        File -> Load File -> select dump.0.rank_*.dump
        OVITO expands the wildcard and merges the data of every rank

    Type mapping:
        - if type_map is given, it is used as is
        - otherwise a globally consistent type_map is built by MPI on the first call
          (types are gathered from every rank, sorted and assigned IDs, so all ranks agree)

    Args:
        output_dir: output directory
        prefix: file name prefix
        box_size: simulation box dimensions (required)
        properties: properties to write
        type_map: mapping from type name to integer (optional; built automatically if absent)
        periodic: periodic boundary conditions
        float_props_names: names of the float properties

    Returns:
        the analysis function

    Examples:
        # Option 1: build type_map automatically (recommended)
        dump_writer = create_distributed_dump_writer(
            output_dir="output/dump",
            box_size=[100.0, 100.0, 100.0],
        )

        # Option 2: give type_map explicitly
        dump_writer = create_distributed_dump_writer(
            output_dir="output/dump",
            box_size=[100.0, 100.0, 100.0],
            type_map={'H': 1, 'V': 2, 'HV': 3},
        )

        post_processor.add_analysis(dump_writer, name="distributed_dump")

    Performance:
        - no gather, so the memory footprint is O(N_local) rather than O(N_total)
        - every rank writes in parallel, with no serial bottleneck
        - suited to large runs with N > 10^6
    """
    if box_size is None:
        raise ValueError("box_size is required for dump writer")

    if properties is None:
        properties = ["id", "type", "x", "y", "z"]
    if periodic is None:
        periodic = [True, True, True]
    if float_props_names is None:
        float_props_names = {}
    # Slot mapping {type: {prop: slot}} of user-declared object properties, installed by
    # the engine through attach_props_mapping(); a user then only lists the property name
    # under "properties". float_props_names stays available as a manual override.
    props_slots = [{}]

    # State variables (kept in lists so the closure can mutate them)
    global_type_map = [type_map]  # used directly when supplied, otherwise built on first call
    initialized = [type_map is not None]  # already initialized when the user supplied type_map

    def _initialize_type_map(pool, comm: MPI.Comm):
        """
        Build a globally consistent type_map on the first call

        Procedure:
        1. every rank collects the types of its local objects
        2. allgather merges all types
        3. IDs are assigned after sorting (so every rank agrees)
        """
        # Collect the local types
        local_objs = pool.get_local_objects(active_only=True)
        local_types = set()
        for obj in local_objs:
            local_types.add(_get_object_type_name(obj, pool))

        # allgather merges all types
        all_types_list = comm.allgather(list(local_types))

        # Merge and sort
        all_types = set()
        for types in all_types_list:
            all_types.update(types)

        # Assign IDs after sorting (so every rank agrees)
        sorted_types = sorted(all_types)
        global_type_map[0] = {t: i + 1 for i, t in enumerate(sorted_types)}
        initialized[0] = True

        if comm.rank == 0 and os.environ.get('PARAOKMC_VERBOSE_DUMP'):
            print(f"[DistributedDump] Auto-generated type_map: {global_type_map[0]}")

    def get_type_id(obj_type: str) -> int:
        """
        Get the type ID

        Raises immediately if the type is not in type_map.
        The user must declare every object type that can appear.
        """
        if global_type_map[0] and obj_type in global_type_map[0]:
            return global_type_map[0][obj_type]

        # Unknown type: raise straight away
        raise KeyError(
            f"Unknown object type '{obj_type}' not in type_map. "
            f"Known types: {list(global_type_map[0].keys()) if global_type_map[0] else []}. "
            f"All object types must be defined in the configuration."
        )

    def get_property_value(obj, prop_name: str, obj_type: str):
        if prop_name == "id":
            return int(obj['id'])
        elif prop_name == "type":
            # The type NAME, as declared in config['objects']: stable across runs and rank
            # counts, imported by OVITO as a named particle type, and mapped straight back by
            # read_dump_frame(). type_map still decides the integer id written as "type_id".
            return str(obj_type)
        elif prop_name == "type_id":
            return get_type_id(str(obj_type))
        elif prop_name in ["x", "y", "z"]:
            return float(obj[prop_name])
        elif prop_name == "size":
            if 'size' in obj.dtype.names:
                return float(obj['size'])
            return 0.0
        elif prop_name == "owner_rank":
            if 'owner_rank' in obj.dtype.names:
                return int(obj['owner_rank'])
            return 0

        if obj_type in float_props_names:
            float_names = float_props_names[obj_type]
            if prop_name in float_names:
                idx = float_names.index(prop_name)
                if 'float_props' in obj.dtype.names and idx < len(obj['float_props']):
                    return float(obj['float_props'][idx])
                return 0.0
        # User-declared object property (objects[type].properties), slot from the engine
        slot = props_slots[0].get(obj_type, {}).get(prop_name)
        if slot is not None and 'float_props' in obj.dtype.names and slot < len(obj['float_props']):
            return float(obj['float_props'][slot])

        if prop_name in obj.dtype.names:
            val = obj[prop_name]
            if isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            else:
                return str(val)

        return 0

    def distributed_dump_writer_func(step: int, sim_time: float, pool, comm: MPI.Comm):
        """Distributed dump writing function"""
        rank = comm.rank
        size = comm.size

        # Build the globally consistent type_map on the first call
        if not initialized[0]:
            _initialize_type_map(pool, comm)

        # Every rank handles only its own objects
        local_objs = pool.get_local_objects(active_only=True)
        n_local = len(local_objs)

        # Collect the total count (for the header; no data is transferred)
        n_total = comm.allreduce(n_local, op=MPI.SUM)

        # Create the output directory (every rank tries; os.makedirs is idempotent)
        os.makedirs(output_dir, exist_ok=True)

        # Every rank writes its own file
        filename = os.path.join(output_dir, f"{prefix}.{step}.rank_{rank}.dump")

        bc_str = " ".join(["pp" if p else "ff" for p in periodic])

        with open(filename, 'w') as f:
            # Header (every file carries a full header; OVITO merges them)
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{step}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{n_local}\n")  # local count
            f.write(f"ITEM: BOX BOUNDS {bc_str}\n")
            f.write(f"0.0 {box_size[0]}\n")
            f.write(f"0.0 {box_size[1]}\n")
            f.write(f"0.0 {box_size[2]}\n")

            props_str = " ".join(properties)
            f.write(f"ITEM: ATOMS {props_str}\n")

            # Write the local objects
            for obj in local_objs:
                obj_type = _get_object_type_name(obj, pool)
                values = [get_property_value(obj, prop, obj_type) for prop in properties]
                formatted = []
                for val in values:
                    if isinstance(val, int):
                        formatted.append(str(val))
                    elif isinstance(val, float):
                        formatted.append(f"{val:.6g}")
                    else:
                        formatted.append(str(val))
                f.write(" ".join(formatted) + "\n")

    def attach_props_mapping(mapping):
        """Engine hook: install the {type: {prop: slot}} mapping of user properties."""
        props_slots[0] = dict(mapping or {})
    distributed_dump_writer_func.attach_props_mapping = attach_props_mapping
    return distributed_dump_writer_func


def create_fast_dump_writer(
    output_dir: str = "dump",
    prefix: str = "dump",
    box_size: List[float] = None,
    properties: List[str] = None,
    type_map: Dict[str, int] = None,
    periodic: List[bool] = None,
    float_props_names: Dict[str, List[str]] = None,
):
    """
    Create a fast dump writer (pre-formatting optimization)

    Every rank pre-formats its data into strings, which are gathered on rank 0 and written.
    This offloads CPU work from rank 0, though the memory pressure remains.

    Suited to medium runs (10^5 < N < 10^6).

    Args:
        as in create_dump_writer()

    Returns:
        the analysis function

    Performance:
        - every rank formats in parallel, offloading CPU work from rank 0
        - a single output file, opened directly by OVITO
        - 2-5x faster than the basic version
    """
    if box_size is None:
        raise ValueError("box_size is required for dump writer")

    if properties is None:
        properties = ["id", "type", "x", "y", "z"]
    if periodic is None:
        periodic = [True, True, True]
    if float_props_names is None:
        float_props_names = {}
    # Slot mapping {type: {prop: slot}} of user-declared object properties, installed by
    # the engine through attach_props_mapping(); a user then only lists the property name
    # under "properties". float_props_names stays available as a manual override.
    props_slots = [{}]

    auto_type_map = [{}]
    next_type_id = [1]

    def get_type_id(obj_type: str) -> int:
        if type_map is not None and obj_type in type_map:
            return type_map[obj_type]
        if obj_type not in auto_type_map[0]:
            auto_type_map[0][obj_type] = next_type_id[0]
            next_type_id[0] += 1
        return auto_type_map[0][obj_type]

    def get_property_value(obj, prop_name: str, obj_type: str):
        if prop_name == "id":
            return int(obj['id'])
        elif prop_name == "type":
            # The type NAME, as declared in config['objects']: stable across runs and rank
            # counts, imported by OVITO as a named particle type, and mapped straight back by
            # read_dump_frame(). type_map still decides the integer id written as "type_id".
            return str(obj_type)
        elif prop_name == "type_id":
            return get_type_id(str(obj_type))
        elif prop_name in ["x", "y", "z"]:
            return float(obj[prop_name])
        elif prop_name == "size":
            if 'size' in obj.dtype.names:
                return float(obj['size'])
            return 0.0
        elif prop_name == "owner_rank":
            if 'owner_rank' in obj.dtype.names:
                return int(obj['owner_rank'])
            return 0

        if obj_type in float_props_names:
            float_names = float_props_names[obj_type]
            if prop_name in float_names:
                idx = float_names.index(prop_name)
                if 'float_props' in obj.dtype.names and idx < len(obj['float_props']):
                    return float(obj['float_props'][idx])
                return 0.0
        # User-declared object property (objects[type].properties), slot from the engine
        slot = props_slots[0].get(obj_type, {}).get(prop_name)
        if slot is not None and 'float_props' in obj.dtype.names and slot < len(obj['float_props']):
            return float(obj['float_props'][slot])

        if prop_name in obj.dtype.names:
            val = obj[prop_name]
            if isinstance(val, (np.floating, float)):
                return float(val)
            elif isinstance(val, (np.integer, int)):
                return int(val)
            else:
                return str(val)

        return 0

    def fast_dump_writer_func(step: int, sim_time: float, pool, comm: MPI.Comm):
        """Fast dump writing function"""
        rank = comm.rank

        # Every rank handles and pre-formats its own objects
        local_objs = pool.get_local_objects(active_only=True)

        # Pre-format into a list of strings
        local_lines = []
        for obj in local_objs:
            obj_type = _get_object_type_name(obj, pool)
            values = [get_property_value(obj, prop, obj_type) for prop in properties]
            formatted = []
            for val in values:
                if isinstance(val, int):
                    formatted.append(str(val))
                elif isinstance(val, float):
                    formatted.append(f"{val:.6g}")
                else:
                    formatted.append(str(val))
            local_lines.append(" ".join(formatted))

        # Gather the pre-formatted strings
        all_lines_list = comm.gather(local_lines, root=0)

        if rank == 0:
            # Merge all lines
            all_lines = []
            for lines in all_lines_list:
                all_lines.extend(lines)

            n_atoms = len(all_lines)

            os.makedirs(output_dir, exist_ok=True)
            filename = os.path.join(output_dir, f"{prefix}.{step}.dump")

            bc_str = " ".join(["pp" if p else "ff" for p in periodic])

            with open(filename, 'w') as f:
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{step}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write(f"{n_atoms}\n")
                f.write(f"ITEM: BOX BOUNDS {bc_str}\n")
                f.write(f"0.0 {box_size[0]}\n")
                f.write(f"0.0 {box_size[1]}\n")
                f.write(f"0.0 {box_size[2]}\n")

                props_str = " ".join(properties)
                f.write(f"ITEM: ATOMS {props_str}\n")

                # Bulk write (faster than line by line)
                f.write("\n".join(all_lines))
                if all_lines:
                    f.write("\n")

    def attach_props_mapping(mapping):
        """Engine hook: install the {type: {prop: slot}} mapping of user properties."""
        props_slots[0] = dict(mapping or {})
    fast_dump_writer_func.attach_props_mapping = attach_props_mapping
    return fast_dump_writer_func
    # A simple test
    comm = MPI.COMM_WORLD
    rank = comm.rank

    print(f"[Rank {rank}] PostProcessor module loaded successfully")

    if rank == 0:
        # Test: create the post-processor
        processor = create_post_processor(comm, interval=10, verbose=True)

        # Add an analysis function
        processor.add_analysis(analysis_object_count, name="object_count")
        processor.add_analysis(analysis_type_distribution, name="type_distribution")

        print(f"[Rank {rank}] PostProcessor test OK")
        print(f"  Registered {len(processor.callbacks)} analysis functions")


# ============================================================================
# Reading a dump back: restart / initial condition from a saved configuration
# ============================================================================

def read_dump_frame(path: str):
    """Read one LAMMPS-style dump file written by any ParaOKMC dump writer.

    Returns (header, columns, rows):
        header  - {'step': int, 'n_atoms': int, 'box': [[lo, hi], ...], 'periodic': [bool]*3}
        columns - the column names of the ITEM: ATOMS line
        rows    - list of lists of strings, one per object (converted by the caller)
    A frame written by the distributed writer holds the objects of one rank only; join the
    `prefix.<step>.rank_*.dump` files of that step to recover the whole system.
    """
    header = {'step': 0, 'n_atoms': 0, 'box': [], 'periodic': [True, True, True]}
    columns, rows = [], []
    with open(path, 'r') as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('ITEM: TIMESTEP'):
            header['step'] = int(lines[i + 1].split()[0]); i += 2
        elif line.startswith('ITEM: NUMBER OF ATOMS'):
            header['n_atoms'] = int(lines[i + 1].split()[0]); i += 2
        elif line.startswith('ITEM: BOX BOUNDS'):
            flags = line.split()[3:6]
            if len(flags) == 3:
                header['periodic'] = [fl == 'pp' for fl in flags]
            header['box'] = [[float(v) for v in lines[i + 1 + k].split()[:2]] for k in range(3)]
            i += 4
        elif line.startswith('ITEM: ATOMS'):
            columns = line.split()[2:]
            i += 1
            while i < len(lines) and lines[i].strip() and not lines[i].startswith('ITEM:'):
                rows.append(lines[i].split()); i += 1
        else:
            i += 1
    return header, columns, rows


def _dump_files_for_frame(path: str, frame):
    """Resolve the files of one frame.

    path   - a single dump file, or a directory written by the dump writers
    frame  - step number, or -1 for the last step found
    Returns a sorted list of files (several when the frame was written per rank).
    """
    import glob
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"ic.file '{path}' is neither a file nor a directory")
    steps = {}
    for f in glob.glob(os.path.join(path, '*.dump')):
        m = re.search(r'\.(\d+)(?:\.rank_\d+)?\.dump$', os.path.basename(f))
        if m:
            steps.setdefault(int(m.group(1)), []).append(f)
    if not steps:
        raise FileNotFoundError(f"no '<prefix>.<step>[.rank_N].dump' files in '{path}'")
    step = max(steps) if frame is None or int(frame) < 0 else int(frame)
    if step not in steps:
        raise FileNotFoundError(f"frame {step} not found in '{path}'; available: {sorted(steps)}")
    return sorted(steps[step])


def create_dump_reader(path: str, frame=-1, id_mode: str = 'keep'):
    """Build an initial_objects_func that restores the objects of a saved dump frame.

    This is what `ic = {"mode": "dump", "file": ..., "frame": ...}` compiles to. The dump
    must carry the columns id, type, x, y, z; `size` and every user-declared property
    (objects[type].properties) are restored when present and left at their declared
    defaults otherwise. The `type` column may hold the type name (what ParaOKMC writes)
    or the integer type id of this configuration.

    Every rank reads the same files and keeps the objects that fall inside its own
    subdomain, exactly like the built-in initial condition. With id_mode='keep' the ids of
    the file are kept (a true restart); with 'renumber' fresh ids are assigned.
    """
    if id_mode not in ('keep', 'renumber'):
        raise ValueError("id_mode must be 'keep' or 'renumber'")

    def initial_objects_func(rank, size, config):
        files = _dump_files_for_frame(path, frame)
        domain_decomp = config['_domain_decomp']
        props_map = (config.get('_props_mapping', {}) or {}).get('float_props', {}) or {}
        int_to_type = {int(k): v for k, v in (config.get('_int_to_type', {}) or {}).items()}
        objects_def = config.get('objects', {}) or {}
        n_float = config['_object_dtype'].fields['float_props'][0].shape[0]

        # declared defaults of every type's float_props (size slot included)
        defaults = {}
        for t, mapping in props_map.items():
            vec = [0.0] * n_float
            props = (objects_def.get(t, {}) or {}).get('properties', {}) or {}
            for name, slot in mapping.items():
                v = props.get(name, 0.0)
                vec[slot] = float(v) if not isinstance(v, str) else 0.0
            defaults[t] = vec

        local, counter = [], 0
        for fpath in files:
            header, cols, rows = read_dump_frame(fpath)
            col = {c: k for k, c in enumerate(cols)}
            for key in ('id', 'type', 'x', 'y', 'z'):
                if key not in col:
                    raise ValueError(f"dump '{fpath}' lacks the column '{key}' (columns: {cols})")
            for r in rows:
                raw_t = r[col['type']]
                t = int_to_type.get(int(raw_t), raw_t) if raw_t.lstrip('-').isdigit() else raw_t
                if t not in objects_def:
                    raise ValueError(f"dump '{fpath}' holds type '{t}', not declared in config['objects']")
                pos = [float(r[col['x']]), float(r[col['y']]), float(r[col['z']])]
                if int(domain_decomp.get_owner_rank(pos)) != rank:
                    continue
                fp = list(defaults.get(t, [0.0] * n_float))
                for name, slot in props_map.get(t, {}).items():
                    if name in col:
                        fp[slot] = float(r[col[name]])
                size_val = float(r[col['size']]) if 'size' in col else float(
                    (objects_def.get(t, {}) or {}).get('size', 0.0))
                counter += 1
                oid = int(r[col['id']]) if id_mode == 'keep' else ((rank << 48) | counter)
                local.append({'id': oid, 'type': t, 'x': pos[0], 'y': pos[1], 'z': pos[2],
                              'size': size_val, 'float_props': fp})
        return local

    return initial_objects_func
