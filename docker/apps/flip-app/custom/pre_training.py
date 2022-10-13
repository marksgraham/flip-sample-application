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

import traceback

from flip import FLIP

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from utils.flip_constants import FlipConstants, ModelStatus, FlipEvents
from utils.utils import Utils


class InitTraining(Controller):
    def __init__(
            self,
            model_id: str,
            min_clients: int = FlipConstants.MIN_CLIENTS,
            task_name: str = FlipConstants.INIT_TRAINING,
            flip: FLIP = FLIP()
    ):
        """The controller that is executed pre-training and is a part of the FLIP training model

        The InitTraining workflow sends a request to the Cental Hub, stating that training has initiated

        Args:
            model_id (str): ID of the model that the training is being performed under.
            min_clients (int, optional): Minimum number of clients. Defaults to 1 for the aggregation to take place with
                successful results.
            task_name (str, optional): Name of the task. Defaults to "init_training".
            flip (FLIP, optional): an instance of the FLIP module.

        Raises:
           ValueError:
            - when the model ID is not a valid UUID.
            - when the minimum number of clients specified is less than 1
        """

        super().__init__()
        self.model_id = model_id
        self.min_clients = min_clients
        self.task_name = task_name
        self.flip = flip

        try:
            if Utils.is_valid_uuid(self.model_id) is False:
                raise ValueError(f"The model ID: {self.model_id} is not a valid UUID")

            if self.min_clients < FlipConstants.MIN_CLIENTS:
                raise ValueError(f"Invalid number of minimum clients specified. {self.min_clients} is less than "
                                 f"{FlipConstants.MIN_CLIENTS} which is the minimum number for a successful aggregation"
                                 )
        except ValueError as e:
            self.flip.update_status(self.model_id, ModelStatus.ERROR)
            raise ValueError(e)

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Initializing InitTraining workflow.")
        engine = fl_ctx.get_engine()
        if not engine:
            self.system_panic("Engine not found. InitTraining exiting.", fl_ctx)
            return

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            if self._check_abort_signal(fl_ctx, abort_signal):
                return

            self.log_info(fl_ctx, "Beginning InitTraining control flow phase.")
            self.init_training(fl_ctx)

            if self._check_abort_signal(fl_ctx, abort_signal):
                return
        except BaseException as e:
            traceback.print_exc()
            error_msg = f"Exception in InitTraining control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Stopping InitTraining controller")
        self.cancel_all_tasks()

    def process_result_of_unknown_task(
            self, client: Client, task_name, client_task_id, result: Shareable, fl_ctx: FLContext
    ) -> None:
        self.log_error(fl_ctx, "Ignoring result from unknown task.")

    def init_training(self, fl_ctx: FLContext):
        try:
            self.log_info(fl_ctx, "Attempting to start the step to initialise training...")
            self.fire_event(FlipEvents.TRAINING_INITIATED, fl_ctx)
        except Exception as e:
            traceback.print_exc()
            self.log_error(fl_ctx, str(e))
            self.system_panic(str(e), fl_ctx)

    def _check_abort_signal(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, f"Abort signal received.")
            self.fire_event(FlipEvents.ABORTED, fl_ctx)
            return True
        return False
