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

import os
import shutil

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext


class CleanupImages(FLComponent):
    def __init__(self, net_id: str):
        """CleanupImages takes place as the final step of the run. All the images used for the training are
        deleted to prevent the build-up of unnecessary files on the storage space

        Args:
            net_id (str): ID of the net containing the images that the model used to perform training on.

        Raises:
           ValueError: when the net ID is empty or none.
        """

        super().__init__()

        if not net_id:
            raise ValueError(f"The net ID: {self.net_id} cannot be empty or None")

    def execute(self, fl_ctx: FLContext):
        try:
            self.log_info(
                fl_ctx, "Images related to the training have been deleted successfully"
            )
        except Exception as e:
            self.log_exception(fl_ctx, str(e))
