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

from paraokmc._version import __version__

# New user interface (recommended)
from paraokmc.user_config import OKMCengine, ConfigValidationError

# Legacy interface (internal use)
from paraokmc.v16_engine import OKMCEngineV16, create_okmc_engine_v16
from paraokmc.event_class import (
    EventBase, SingleObjectEvent, PairObjectEvent, ZeroObjectEvent,
    EventRegistry, compile_event_config,
    get_physics_param, get_physics_field, get_physics_field_cached,
    ArrheniusMigration, DiffusionCapture, Annihilation,
    delete_obj, change_obj, new_obj,
)
from paraokmc.event_compiler import EventCompiler, compile_events, merge_event_config
from paraokmc.unified_physics import (
    UnifiedPhysics, ExternalSolver, create_unified_physics,
    MODE_TWO_WAY, MODE_OUTWARD, MODE_INWARD,
)
from paraokmc.config_system import create_initial_objects_func
from paraokmc.jit_rate_engine import RateDeps
from paraokmc.helpers import (
    TemperatureScheduler, KB_EV_K,
    deterministic_random, deterministic_randint, deterministic_choice,
)
from paraokmc.random_generator import RandomGenerator
