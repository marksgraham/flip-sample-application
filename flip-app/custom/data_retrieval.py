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

from flip import FLIP

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.apis.event_type import EventType
from utils.flip_constants import FlipConstants
from utils.utils import Utils


class DataRetrieval(Executor):
    def __init__(
        self,
        project_id: str,
        net_id: str,
        retrieval_task_name: str = FlipConstants.RETRIEVE_IMAGES,
        flip: FLIP = FLIP(),
    ):
        """DataRetrieval takes place before the training. All the images that will be used as part the training are
        downloaded to a specified and accessible storage space

        Args:
            project_id (str): ID of the project containing the images that the model will use to perform training.
            net_id (str): ID of the net containing the images that the model will use to perform training.
            retrieval_task_name: (str, optional): Task name for retrieval task. Defaults to "retrieve_images".

        Raises:
           ValueError:
            - when the project ID is not a valid UUID.
            - when the net ID is empty or None.
        """

        super().__init__()
        self.project_id = project_id
        self.net_id = net_id
        self.retrieval_task_name = retrieval_task_name
        self.flip = flip

        if Utils.is_valid_uuid(project_id) is False:
            raise ValueError(f"The project ID: {self.project_id} is not a valid UUID")

        if not net_id:
            raise ValueError(f"The net ID: {self.net_id} cannot be empty or None")

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        try:
            if task_name == self.retrieval_task_name:
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                path = "/path/to/data"
                self.log_info(
                    fl_ctx,
                    f"Images related to the training have been downloaded successfully at path: {str(path)}",
                )

                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                return make_reply(ReturnCode.OK)
            else:
                return make_reply(ReturnCode.TASK_UNKNOWN)
        except Exception as e:
            self.log_exception(fl_ctx, str(e))
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)
