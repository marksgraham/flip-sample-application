import ast
import json
import os
import requests

from pathlib import Path
from pandas import DataFrame
from utils.utils import Utils

from utils.flip_constants import FlipConstants, ModelStatus


class FLIP:

    def get_dataframe(self, project_id: str, query: str) -> DataFrame:
        """Calls the data-access-api in FLIP to return a dataframe.

        Returns:
            DataFrame: pandas dataframe
        """

    def get_data(self, project_id: str, net_id: str) -> Path:
        """Calls the imaging-service in FLIP to return a filepath that contains images and labels downloaded from XNAT.

        Returns:
            Path: path of data
        """

    def update_status(self, model_id: str, new_model_status: ModelStatus):
        """Updates the Central Hub model status
        """
