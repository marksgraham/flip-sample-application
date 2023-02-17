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

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from flip import FLIP
from monai.data import DataLoader, Dataset
from monai.inferers import sliding_window_inference
from monai.metrics import compute_meandice
from monai.transforms import (
    AddChannelD,
    AddCoordinateChannelsD,
    CastToTypeD,
    Compose,
    ConcatItemsD,
    DeleteItemsD,
    LoadImageD,
    ScaleIntensityRangeD,
    SplitChannelD,
    ToTensorD,
)
from nvflare.apis.dxo import DXO, DataKind, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from simple_network import SimpleNetwork


class FLIP_VALIDATOR(Executor):
    def __init__(self, validate_task_name=AppConstants.TASK_VALIDATION, project_id="", query=""):
        super(FLIP_VALIDATOR, self).__init__()

        self._validate_task_name = validate_task_name

        # Setup the model
        self.model = SimpleNetwork()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        # NB val transforms differ from the train transforms. No random affine augmentation is applied and the data is
        # not cropped into patches.
        self.val_transforms = Compose(
            [
                LoadImageD(keys=["img", "seg"], reader="NiBabelReader", as_closest_canonical=False),
                AddChannelD(keys=["img", "seg"]),
                AddCoordinateChannelsD(keys=["img"], spatial_dims=(0, 1, 2)),
                SplitChannelD(keys=["img"]),
                ScaleIntensityRangeD(keys=["img_0"], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
                ConcatItemsD(keys=["img_0", "img_1", "img_2", "img_3"], name="img"),
                DeleteItemsD(
                    keys=["img_0", "img_1", "img_2", "img_3"],
                ),
                CastToTypeD(keys=["img"], dtype=np.float32),
                ToTensorD(keys=["img", "seg"]),
            ]
        )

        # Setup the training dataset
        self.flip = FLIP()
        self.project_id = project_id
        self.query = query
        self.dataframe = self.flip.get_dataframe(self.project_id, self.query)

    def get_image_and_label_list(self, dataframe, val_split=0.5):
        """Returns a list of dicts, each dict containing the path to an image and its corresponding label."""

        datalist = []
        # loop over each accession id in the val set
        for accession_id in dataframe["accession_id"]:
            try:
                image_data_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)
            except Exception as e:
                print(f"Could not get image data folder path for {accession_id}:")
                print(f"{e=}")
                print(f"{type(e)=}")
                print(f"{e.args=}")
                continue

            accession_folder_path = image_data_folder_path

            all_images = list(Path(accession_folder_path).rglob("images/*.nii.gz"))
            first_n=18
            print(f'Limiting to {first_n} images:')
            all_images=all_images[:first_n]
            this_accession_matches = 0
            print(f"Total base CT count found for accession_id {accession_id}: {len(all_images)}")
            for img in all_images:
                seg = str(img).replace('images/', 'labels/')

                if not Path(seg).exists():
                    print(f"No matching lesion mask for {img}.")
                    continue

                try:
                    img_header = nib.load(str(img))
                except nib.filebasedimages.ImageFileError as err:
                    print(f"Problem loading header of base image {str(img)}.")
                    print(f"{err=}")
                    print(f"{type(err)=}")
                    print(f"{err.args=}")
                    continue

                try:
                    seg_header = nib.load(seg)
                except nib.filebasedimages.ImageFileError as err:
                    print(f"Problem loading header of segmentation {str(seg)}.")
                    print(f"{err=}")
                    print(f"{type(err)=}")
                    print(f"{err.args=}")
                    continue

                # check is 3D and at least 128x128x128 in size and seg is the same
                if len(img_header.shape) != 3:
                    print(f"Image has other than 3 dimensions (it has {len(img_header.shape)}.)")
                    continue
                elif any([dim < 128 for dim in img_header.shape]):
                    print(f"Image has one or more dimensions <128: ({img_header.shape}).")
                    continue
                elif any([img_dim != seg_dim for img_dim, seg_dim in zip(img_header.shape, seg_header.shape)]):
                    print(
                        f"Image dimensions ({img_header.shape}) do not match segmentation dimensions ({seg_header.shape}).")
                    continue
                else:
                    datalist.append({"img": str(img), "seg": seg})
                    print(f"Matching base image and segmentation added.")
                    this_accession_matches += 1
            print(f"Added {this_accession_matches} matched image + segmentation pairs for {accession_id}.")
        print(f"Found {len(datalist)} files in train")

        # split into the training and testing data
        train_datalist, val_datalist = np.split(datalist, [int((1 - val_split) * len(datalist))])

        return val_datalist

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
            dxo = from_shareable(shareable)

            # Ensure data_kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                self.log_exception(
                    fl_ctx,
                    f"DXO is of type {dxo.data_kind} but expected type WEIGHTS.",
                )
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Extract weights and ensure they are tensor.
            model_owner = shareable.get_header(AppConstants.MODEL_OWNER, "?")
            weights = {k: torch.as_tensor(v, device=self.device) for k, v in dxo.data.items()}

            # Get validation accuracy
            val_accuracy = self.do_validation(fl_ctx, weights, abort_signal)
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

        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def do_validation(self, fl_ctx, weights, abort_signal):
        self.model.load_state_dict(weights)
        self.model.eval()
        total_mean_dice = 0
        num_images = 0
        print(len(self.test_loader))
        with torch.no_grad():
            for i, batch in enumerate(self.test_loader):
                if abort_signal.triggered:
                    return 0

                images, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
                # perform sliding window inference to get a prediction for the whole volume.
                output_logits = sliding_window_inference(
                    images,
                    sw_batch_size=2,
                    roi_size=(128, 128, 128),
                    predictor=self.model,
                    overlap=0.25,
                    do_sigmoid=False,
                )
                output = torch.sigmoid(output_logits)
                metric = compute_meandice(output, labels, include_background=False).cpu().numpy()

                total_mean_dice += metric.sum()
                num_images += images.size()[0]
                print(f"Validator Iteration: {i}, Metric: {total_mean_dice}, Num Images: {num_images}")

            metric = total_mean_dice / float(num_images)
            print(f"Validator Iteration finished: {i}, Metric: {total_mean_dice/num_images}")

            self.flip.send_metrics_value(label="VAL_LOSS", value=metric, fl_ctx=fl_ctx)

        return metric
