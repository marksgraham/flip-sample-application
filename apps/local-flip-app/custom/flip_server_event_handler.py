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

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.app_event_type import AppEventType

from flip import FLIP
from persist_and_cleanup import PersistToS3AndCleanup
from utils.flip_constants import ModelStatus, FlipEvents
from utils.utils import Utils
from validation_json_generator import ValidationJsonGenerator


class ServerEventHandler(FLComponent):
    """ ServerEventHandler is a generic component that handles system events triggered by nvflare
        or custom flip events. It executes logic inside its own event handler but may also call
        other component's event handlers directly to overcome the non-deterministic order
        in which nvflare handles events i.e handling ValidationJsonGenerator component events.

        Args:
            model_id (string, not required)
            validation_json_generator_id (string, not required)
            persist_and_cleanup_id (string, not required)
            flip (object, not required)
        Raises:
            ValueError: when model ID is not a valid UUID
    """

    def __init__(self,
                 model_id: str = "",
                 validation_json_generator_id: str = "json_generator",
                 persist_and_cleanup_id: str = "persist_and_cleanup",
                 flip: FLIP = FLIP()
                 ):
        super(ServerEventHandler, self).__init__()

        self.model_id = model_id
        self.validation_json_generator_id = validation_json_generator_id
        self.validation_json_generator = None
        self.persist_and_cleanup_id = persist_and_cleanup_id
        self.persist_and_cleanup = None
        self.flip = flip

        self.final_status = None

        if Utils.is_valid_uuid(self.model_id) is False:
            raise ValueError(f"The model ID: {self.model_id} is not a valid UUID")

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        self.__set_dependencies(fl_ctx)

        self.validation_json_generator.handle_validation_events(event_type, fl_ctx)

        if event_type == EventType.FATAL_SYSTEM_ERROR:
            self.log_error(fl_ctx, "Fatal system error event received")
            self.final_status = ModelStatus.ERROR

        elif event_type == FlipEvents.TRAINING_INITIATED:
            self.log_info(fl_ctx, "Training initiated event received")

        elif event_type == AppEventType.INITIAL_MODEL_LOADED:
            self.log_info(fl_ctx, "Initial model loaded event received")

        elif event_type == AppEventType.TRAINING_STARTED:
            self.log_info(fl_ctx, "Training started event received")

        elif event_type == AppEventType.TRAINING_FINISHED:
            self.log_info(fl_ctx, "Training finished event received")

        elif event_type == FlipEvents.RESULTS_UPLOAD_COMPLETED:
            self.log_info(fl_ctx, "Results upload completed event received")

        elif event_type == FlipEvents.ABORTED:
            self.log_info(fl_ctx, "Aborted event received")
            self.final_status = ModelStatus.STOPPED

        elif event_type == EventType.START_RUN:
            self.log_info(fl_ctx, "Start run event received")

        elif event_type == EventType.END_RUN:
            self.log_info(fl_ctx, "End run event received")

            try:
                self.persist_and_cleanup.execute(fl_ctx)

                if self.final_status != ModelStatus.STOPPED and self.final_status != ModelStatus.ERROR:
                    self.final_status = ModelStatus.RESULTS_UPLOADED
            except Exception:
                self.final_status = ModelStatus.ERROR

    def __set_dependencies(self, fl_ctx: FLContext):
        if self.validation_json_generator is None:
            engine = fl_ctx.get_engine()
            self.validation_json_generator = engine.get_component(self.validation_json_generator_id)

            if self.validation_json_generator is None \
                    or not isinstance(self.validation_json_generator, ValidationJsonGenerator):
                self.system_panic(
                    f"'validation_json_generator_id' component must be ValidationJsonGenerator. "
                    f"But got: {type(self.validation_json_generator)}",
                    fl_ctx,
                )
                return

        if self.persist_and_cleanup is None:
            engine = fl_ctx.get_engine()
            self.persist_and_cleanup = engine.get_component(self.persist_and_cleanup_id)

            if self.persist_and_cleanup is None \
                    or not isinstance(self.persist_and_cleanup, PersistToS3AndCleanup):
                self.system_panic(
                    f"'persist_and_cleanup_id' component must be PersistToS3AndCleanup. "
                    f"But got: {type(self.persist_and_cleanup)}",
                    fl_ctx,
                )
                return
