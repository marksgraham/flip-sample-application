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

import ast
import json
import os
import requests

from pathlib import Path
from pandas import DataFrame
from typing import List
from utils.utils import Utils

from utils.flip_constants import FlipEvents, ModelStatus

from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.fl_constant import FLContextKey, EventScope, FedEventHeader
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable


class FLIP:
    def get_dataframe(self, project_id: str, query: str) -> DataFrame:
        """Calls the FLIP service to return a dataframe.

        Returns:
            DataFrame: pandas dataframe
        """

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
            raise TypeError("expect label to be string, but got {}".format(type(label)))

        if not isinstance(fl_ctx, FLContext):
            raise TypeError("expect fl_ctx to be FLContext, but got {}".format(type(fl_ctx)))

        engine = fl_ctx.get_engine()
        if engine is None:
            print("Error: no engine in fl_ctx, cannot fire metrics event")
            return

        dxo = DXO(data_kind=DataKind.METRICS, data={
            'label': label,
            'value': value
        })
        event_data = dxo.to_shareable()

        fl_ctx.set_prop(FLContextKey.EVENT_DATA, event_data, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.EVENT_SCOPE, value=EventScope.FEDERATION, private=True, sticky=False)
        fl_ctx.set_prop(FLContextKey.EVENT_ORIGIN, "flip_client", private=True, sticky=False)

        engine.fire_event(FlipEvents.SEND_RESULT, fl_ctx)


    def handle_metrics_event(self, event_data: Shareable, global_round: int, model_id: str):
        """Use on the server to handle metrics data events raised by clients
        """
