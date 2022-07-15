# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
        self.model_dir: str = f"{cwd}/run_1"
        self.bucket_name: str = (
            f"flip-uploaded-federated-data-bucket-{os.environ.get('ENVIRONMENT')}"
        )

        self.flip = flip

        if Utils.is_valid_uuid(self.model_id) is False:
            self.flip.update_status(self.model_id, ModelStatus.ERROR)
            raise ValueError(f"The model ID: {self.model_id} is not a valid UUID")

    def execute(self, fl_ctx: FLContext):
        try:
            self.log_info(fl_ctx, "Initializing PersistToS3AndCleanup")
            engine = fl_ctx.get_engine()
            if not engine:
                self.system_panic(
                    "Engine not found. PersistToS3AndCleanup exiting.", fl_ctx
                )
                return

            self.model_persistor: PTFileModelPersistor = engine.get_component(
                self.persistor_id
            )
            if self.model_persistor is None or not isinstance(
                self.model_persistor, PTFileModelPersistor
            ):
                self.system_panic(
                    f"'persistor_id' component must be PTFileModelPersistor. But got: {type(self.model_persistor)}",
                    fl_ctx,
                )
                return

            self.log_info(fl_ctx, "Beginning PersistToS3AndCleanup")
            self.model_inventory = self.model_persistor.get_model_inventory(fl_ctx)

            if (self.model_inventory.get(PTConstants.PTFileModelName) is not None) and (
                PTConstants.PTFileModelName in self.model_inventory
            ):

                self.model_dir = (
                    self.model_inventory[PTConstants.PTFileModelName].location.split(
                        f"run_1"
                    )[0]
                    + f"run_1"
                )

                self.log_info(
                    fl_ctx, "Location of the final aggregated model obtained."
                )
            else:
                self.log_warning(
                    fl_ctx,
                    "Unable to retrieve the details of the aggregated model. "
                    "Will attempt to zip everything within the final run using a manual path.",
                )

            self.fire_event(FlipEvents.RESULTS_UPLOAD_STARTED, fl_ctx)

            self.upload_results_to_s3_bucket(self.model_dir, fl_ctx)

            self.fire_event(FlipEvents.RESULTS_UPLOAD_COMPLETED, fl_ctx)

            self.cleanup(fl_ctx)

            self.log_info(fl_ctx, "PersistToS3AndCleanup completed")

        except BaseException as e:
            traceback.print_exc()
            error_msg = f"Exception in PersistToS3AndCleanup control_flow: {e}"
            self.log_exception(fl_ctx, error_msg)

    def upload_results_to_s3_bucket(self, source_path: str, fl_ctx: FLContext):
        try:
            self.log_info(
                fl_ctx,
                "Attempting to upload the final aggregated model to the s3 bucket...",
            )

            run_dir = os.path.join(cwd, "run_1")
            app_server_path = os.path.join(run_dir, "app_server")

            fl_global_model_filepath = os.path.join(
                app_server_path, "FL_global_model.pt"
            )
            trainer_path = os.path.join(app_server_path, "custom", "trainer.py")
            validator_path = os.path.join(app_server_path, "custom", "validator.py")

            if os.path.isfile(fl_global_model_filepath):
                shutil.move(fl_global_model_filepath, run_dir)

            if os.path.isfile(trainer_path):
                shutil.move(trainer_path, run_dir)

            if os.path.isfile(validator_path):
                shutil.move(validator_path, run_dir)

            if os.path.isdir(run_dir):
                shutil.rmtree(os.path.join(run_dir, "app_server"))

            self.log_info(fl_ctx, "Zipping the final model and the reports...")
            zip_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_path = os.path.join(cwd, "save", zip_name)
            shutil.make_archive(zip_path, "zip", os.path.abspath(source_path))

            self.log_info(fl_ctx, "Uploading zip file...")
            bucket_zip_path = f"{self.model_id}/{zip_name}"

            s3_client = boto3.client("s3")
            s3_client.upload_file(
                zip_path + ".zip", self.bucket_name, bucket_zip_path + ".zip"
            )

            self.log_info(fl_ctx, "Upload to the s3 bucket successful")
        except FileNotFoundError as e:
            self.log_error(fl_ctx, f"File or directory: {e.filename} does not exist")
            self.log_error(fl_ctx, str(e))
            self.fire_event(FlipEvents.RESULTS_UPLOAD_ERROR, fl_ctx)
        except Exception as e:
            self.log_error(fl_ctx, "Upload to the s3 bucket failed")
            self.log_error(fl_ctx, str(e))
            self.fire_event(FlipEvents.RESULTS_UPLOAD_ERROR, fl_ctx)

    def cleanup(self, fl_ctx: FLContext):
        try:
            self.log_info(
                fl_ctx,
                "Attempting to delete the zip file containing the final aggregated run on disk...",
            )

            run_dir = os.path.join(cwd, "run_1")
            save_dir = os.path.join(cwd, "save")

            self.log_info(fl_ctx, cwd)

            for filename in os.listdir(run_dir):
                file_path = os.path.join(run_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    self.log_error(f"Failed to delete {file_path}. Reason: {e}")

            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)

            self.log_info(fl_ctx, "Zip file has been deleted successfully")

        except Exception as e:
            self.log_error(
                fl_ctx, "Cleanup step to delete the images used for training failed"
            )
            self.log_error(fl_ctx, str(e))
            self.fire_event(FlipEvents.CLEANUP_ERROR, fl_ctx)
