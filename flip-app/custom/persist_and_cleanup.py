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

from pathlib import Path

from nvflare.apis.fl_component import FLComponent

from utils.flip_constants import FlipEvents

from pt_constants import PTConstants

import traceback
import os
import boto3
import shutil

from datetime import datetime

from nvflare.app_common.app_constant import AppConstants
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.pt.pt_file_model_persistor import PTFileModelPersistor
from utils.flip_constants import FlipConstants, ModelStatus
from utils.utils import Utils
from flip import FLIP

cwd = str(Path.cwd())  # Server dir


class PersistToS3AndCleanup(FLComponent):
    def __init__(
        self,
        model_id: str,
        persistor_id: str = AppConstants.DEFAULT_PERSISTOR_ID,
        flip: FLIP = FLIP(),
    ):
        """The component that is executed post training and is a part of the FLIP training model

        The PersistToS3AndCleanup workflow saves the aggregated model (once training has finished) to an S3 bucket, and
        then deletes files created as part of the run

        Args:
            model_id (str): ID of the model that the training is being performed under.
            persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".

        Raises:
           ValueError:
            - when the model ID is not a valid UUID.

            FileNotFoundError: boto3 error for when the zip file does not exist.
        """

        super().__init__()
        self.model_id = model_id
        self.persistor_id = persistor_id
        self.model_persistor = None
        self.model_inventory: dict = {}
        
        self.flip = flip

        if Utils.is_valid_uuid(self.model_id) is False:
            self.flip.update_status(self.model_id, ModelStatus.ERROR)
            raise ValueError(f"The model ID: {self.model_id} is not a valid UUID")

    def execute(self, fl_ctx: FLContext):
        try:
            self.log_info(fl_ctx, "Initializing PersistToS3AndCleanup")

            self.log_info(fl_ctx, "Beginning PersistToS3AndCleanup")

            self.log_info(fl_ctx, "PersistToS3AndCleanup completed")

        except BaseException as e:
            traceback.print_exc()
            error_msg = f"Exception in PersistToS3AndCleanup control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)