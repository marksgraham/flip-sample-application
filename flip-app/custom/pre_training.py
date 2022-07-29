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


class DataRetrieval(Controller):
    def __init__(
        self,
        model_id: str,
        min_clients: int = FlipConstants.MIN_CLIENTS,
        retrieval_task_name: str = FlipConstants.RETRIEVE_IMAGES,
        flip: FLIP = FLIP(),
    ):
        """The controller that is executed pre-training and is a part of the FLIP training model

        The DataRetrieval workflow sends a request to each of the participating clients to retrieve the images that it
            will use for training

        Args:
            model_id (str): ID of the model that the training is being performed under.
            min_clients (int, optional): Minimum number of clients. Defaults to 2 for the aggregation to take place with
                successful results.
            retrieval_task_name (str, optional): Name of the retrieval task. Defaults to "retrieve_images".

        Raises:
           ValueError:
            - when the model ID is not a valid UUID.
            - when the minimum number of clients specified is less than 2
        """

        super().__init__()
        self.model_id = model_id
        self.min_clients = min_clients
        self.retrieval_task_name = retrieval_task_name
        self.flip = flip

        try:
            if Utils.is_valid_uuid(self.model_id) is False:
                raise ValueError(f"The model ID: {self.model_id} is not a valid UUID")

            if self.min_clients < FlipConstants.MIN_CLIENTS:
                raise ValueError(
                    f"Invalid number of minimum clients specified. {self.min_clients} is less than "
                    f"{FlipConstants.MIN_CLIENTS} which is the minimum number for a successful aggregation"
                )
        except ValueError as e:
            self.flip.update_status(self.model_id, ModelStatus.ERROR)
            raise ValueError(e)

    def start_controller(self, fl_ctx: FLContext):
        self.log_info(fl_ctx, "Initializing DataRetrieval workflow.")
        engine = fl_ctx.get_engine()
        if not engine:
            self.system_panic("Engine not found. DataRetrieval exiting.", fl_ctx)
            return

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        try:
            if self._check_abort_signal(fl_ctx, abort_signal):
                return

            self.log_info(fl_ctx, "Beginning DataRetrieval control flow phase.")
            self.retrieve_images(fl_ctx)

            if self._check_abort_signal(fl_ctx, abort_signal):
                return
        except BaseException as e:
            traceback.print_exc()
            error_msg = f"Exception in DataRetrieval control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)
            self.system_panic(str(e), fl_ctx)

    def stop_controller(self, fl_ctx: FLContext) -> None:
        self.log_info(fl_ctx, "Stopping DataRetrieval controller")
        self.cancel_all_tasks()

    def process_result_of_unknown_task(
        self,
        client: Client,
        task_name,
        client_task_id,
        result: Shareable,
        fl_ctx: FLContext,
    ) -> None:
        self.log_error(fl_ctx, "Ignoring result from unknown task.")

    def retrieve_images(self, fl_ctx: FLContext):
        try:
            self.log_info(
                fl_ctx,
                "Attempting to start the step to retrieve and download the images required for training...",
            )
            shareable = Shareable()

            retrieval_task = Task(
                name=self.retrieval_task_name, data=shareable, props={}
            )

            self.fire_event(FlipEvents.DATA_RETRIEVAL_STARTED, fl_ctx)

            self.broadcast_and_wait(
                task=retrieval_task, 
                min_responses=self.min_clients, 
                fl_ctx=fl_ctx
            )

            self.log_info(fl_ctx, "Retrieval of images step successful")

        except Exception as e:
            traceback.print_exc()
            self.log_error(fl_ctx, "Retrieval of images step failed")
            self.log_error(fl_ctx, str(e))
            self.system_panic(str(e), fl_ctx)

    def _check_abort_signal(self, fl_ctx, abort_signal: Signal):
        if abort_signal.triggered:
            self._phase = AppConstants.PHASE_FINISHED
            self.log_info(fl_ctx, f"Abort signal received. Exiting at round {self._current_round}.")
            self.fire_event(FlipEvents.ABORTED, fl_ctx)
            return True
        return False
