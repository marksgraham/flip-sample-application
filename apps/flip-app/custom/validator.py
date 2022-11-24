# Copyright (c) 2021, NVIDIA CORPORATION.
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
from flip import FLIP

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam

from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
from monai.transforms import (Compose, LoadImageD, AddChannelD, AddCoordinateChannelsD, Rand3DElasticD, SplitChannelD,
                              DeleteItemsD, ScaleIntensityRangeD, ConcatItemsD, RandSpatialCropD, ToTensorD,
                              CastToTypeD)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice

from nvflare.apis.dxo import from_shareable, DataKind, DXO
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from simple_network import SimpleNetwork


class FLIP_VALIDATOR(Executor):
    def __init__(self,         validate_task_name=AppConstants.TASK_VALIDATION,
        project_id="",
        query=""):
        super(FLIP_VALIDATOR, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork()
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.model.to(self.device)

        # NB val transforms differ from the train transforms. No random affine augmentation is applied and the data is
        # not cropped into patches.
        self.val_transforms = Compose(
            [LoadImageD(keys=['img', 'seg'], reader='NiBabelReader', as_closest_canonical=False),
             AddChannelD(keys=['img', 'seg']),
             AddCoordinateChannelsD(keys=['img'], spatial_dims=(0, 1, 2)),
             SplitChannelD(keys=['img']),
             ScaleIntensityRangeD(keys=['img_0'], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
             ConcatItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], name='img'),
             DeleteItemsD(keys=['img_0', 'img_1', 'img_2', 'img_3'], ),
             CastToTypeD(keys=['img'], dtype=np.float32),
             ToTensorD(keys=['img', 'seg'])
             ])

        

        # Setup the training dataset
        self.flip = FLIP()
        self.project_id = project_id
        self.query = query
        self.dataframe = self.flip.get_dataframe(self.project_id, self.query)

    def get_image_and_label_list(self, dataframe, val_split=0.1):
            '''Returns a list of dicts, each dict containing the path to an image and its corresponding label.
            '''
            # split into the training and testing data
            train_dataframe, val_dataframe =  np.split(dataframe, [int((1-val_split)*len(dataframe))])
            image_and_label_files = []
            # loop over each accession id in the train set
            for accession_id in val_dataframe['accession_id']:
                try:
                    accession_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)
                    # search for all .nii in the folder and check to see if they have a corresponding label
                    all_images = accession_folder_path.rglob('*.nii*')
                    for image in all_images:
                        stem = str(image.stem).replace('.gz','').replace('.nii','') 
                        # after data enrichment the segmentation will be named something like filepath_label like this
                        #label_path = accession_folder_path / f'{stem}_label.nii'
                        # we aren't doing data enrichment so we'll just set the label to the image here
                        label_path = image
                        if label_path.exists():
                            image_and_label_files.append(
                                {'img': str(image),
                                'seg': str(label_path)})
                        else:
                            num_unpaired += 1
                except:
                    pass   
            print(f'Found {len(image_and_label_files)} files in val')
            return image_and_label_files

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:
        
        test_dict = self.get_image_and_label_list(self.dataframe)
        self._test_dataset = Dataset(test_dict, transform=self.val_transforms)
        self.test_loader = DataLoader(self._test_dataset, batch_size=1, shuffle=False)

        if task_name == self._validate_task_name:
            model_owner = "?"
            try:
                try:
                    dxo = from_shareable(shareable)
                except:
                    self.log_error(fl_ctx, "Error in extracting dxo from shareable.")
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Ensure data_kind is weights.
                if not dxo.data_kind == DataKind.WEIGHTS:
                    self.log_exception(
                        fl_ctx,
                        f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.",
                    )
                    return make_reply(ReturnCode.BAD_TASK_DATA)

                # Extract weights and ensure they are tensor.
                model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
                weights = {
                    k: torch.as_tensor(v, device=self.device)
                    for k, v in dxo.data.items()
                }

                # Get validation accuracy
                val_accuracy = self.do_validation(weights, abort_signal)
                if abort_signal.triggered:
                    return make_reply(ReturnCode.TASK_ABORTED)

                self.log_info(
                    fl_ctx,
                    f"Accuracy when validating {model_owner}'s model on"
                    f" {fl_ctx.get_identity_name()}"
                    f"s data: {val_accuracy}",
                )

                dxo = DXO(data_kind=DataKind.METRICS, data={"val_acc": val_accuracy})
                return dxo.to_shareable()
            except:
                self.log_exception(
                    fl_ctx, f"Exception in validating model from {model_owner}"
                )
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, weights, abort_signal):
        self.model.load_state_dict(weights)
        self.model.eval()
        total_mean_dice = 0
        num_images = 0
        print(len(self.test_loader))
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = batch['img'].to(self.device), batch['seg'].to(self.device)
                # perform sliding window inference to get a prediction for the whole volume.
                output_logits = sliding_window_inference(images,
                                                         sw_batch_size=2,
                                                         roi_size=(128, 128, 128),
                                                         predictor=self.model,
                                                         overlap=0.25,
                                                         do_sigmoid=False)
                output = torch.sigmoid(output_logits)
                metric = compute_meandice(output, labels, include_background=False).cpu().numpy()

                total_mean_dice += metric.sum()
                num_images += images.size()[0]
                print(f'Validator Iteration: {i}, Metric: {total_mean_dice}, Num Images: {num_images}')


            metric = total_mean_dice/ float(num_images)

        return metric