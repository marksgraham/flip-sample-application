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

import logging
import json
import ast
import os.path
from pathlib import Path
from typing import List

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.fl_constant import FLContextKey, EventScope
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from pandas import DataFrame

from utils.flip_constants import FlipEvents, ModelStatus


class FLIP:
    def __init__(self):
        self._name = self.__class__.__name__
        self.logger = logging.getLogger(self._name)

    def get_dataframe(self, project_id: str, query: str) -> DataFrame:
        """In production, this method calls the FLIP service to return a dataframe. Within this sample app,
        a static sample response is provided instead.

        Returns:
            DataFrame: pandas dataframe
        """

        script_path = os.path.realpath(os.path.dirname(__file__))
        sample_json_filepath = os.path.join(script_path, "sample_get_dataframe_response.json")

        self.logger.info(f"Retrieving sample dataframe from {sample_json_filepath}")

        if not os.path.isfile(sample_json_filepath):
            self.logger.error("No sample dataframe json file could be found!")
            raise FileNotFoundError(sample_json_filepath);

        with open(sample_json_filepath) as outfile:
            response = json.load(outfile)

            df = DataFrame(data=response)

            self.logger.info("Successfully parsed sample dataframe")

            return df


    def get_by_accession_number(self, project_id: str, accession_id: str) -> Path:
        """Calls the FLIP service to return a filepath that contains images downloaded from XNAT based
           on the accession number

        Returns:
            Path: path to data
        """

    def add_resource(self, project_id: str, accession_id: str, scan_id: str, resource_id: str, files: List[str]):
        """Calls the FLIP service to upload image(s) to XNAT based on the accession number, scan ID, and resource ID
        """

    def update_status(self, model_id: str, new_model_status: ModelStatus):
        """INTENDED FOR INTERNAL USE ONLY. NOT TO BE CALLED BY THE TRAINER.
        """

    def send_metrics_value(self, label: str, value: float, fl_ctx: FLContext):
        """Raises a federated event containing the passed in metrics data (label, value)
        """
        if not isinstance(label, str):
            raise TypeError("Expect label to be string, but got {}".format(type(label)))

        if not isinstance(fl_ctx, FLContext):
            raise TypeError("Expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))

        engine = fl_ctx.get_engine()
        if engine is None:
            self.logger.error("Error: no engine in fl_ctx, cannot fire metrics event")
            return

        self.logger.info("Attempting to fire metrics event...")

        dxo = DXO(data_kind=DataKind.METRICS, data={
            'label': label,
            'value': value
        })
        event_data = dxo.to_shareable()

        fl_ctx.set_prop(FLContextKey.EVENT_DATA, event_data, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, value=EventScope.FEDERATION, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.EVENT_ORIGIN, "flip_client", private=True, sticky=False)

        engine.fire_event(FlipEvents.SEND_RESULT, fl_ctx)

        self.logger.info("Successfully fired metrics event")

    def handle_metrics_event(self, event_data: Shareable, global_round: int, model_id: str):
        """INTENDED FOR INTERNAL USE ONLY. NOT TO BE CALLED BY THE TRAINER.
        """
