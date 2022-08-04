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

from utils.flip_constants import FlipConstants, ModelStatus


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
