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

from enum import Enum


class FlipConstants(object):
    CLEANUP: str = "cleanup"
    MIN_CLIENTS: int = 1
    INIT_TRAINING: str = "init_training"


class FlipEvents(object):
    TRAINING_INITIATED = "_training_initiated"
    RESULTS_UPLOAD_STARTED = "_results_upload_started"
    RESULTS_UPLOAD_COMPLETED = "_results_upload_completed"
    ABORTED = "_aborted"
    SEND_RESULT = "_send_result"


class ModelStatus(str, Enum):
    PENDING = "PENDING",
    INITIATED = "INITIATED",
    PREPARED = "PREPARED",
    TRAINING_STARTED = "TRAINING_STARTED",
    RESULTS_UPLOADED = "RESULTS_UPLOADED",
    ERROR = "ERROR",
    STOPPED = "STOPPED"


class FlipMetricsLabel(str, Enum):
    LOSS_FUNCTION = "LOSS_FUNCTION"
    DL_RESULT = "DL_RESULT"
    AVERAGE_SCORE = "AVERAGE_SCORE"
