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

import os.path
from pathlib import Path
import json

import nibabel as nib
import numpy as np
import torch
from flip import FLIP
from monai.data import DataLoader, Dataset
from monai.losses import DiceLoss
from monai.transforms import (
    AddChannelD,
    AddCoordinateChannelsD,
    Compose,
    ConcatItemsD,
    DeleteItemsD,
    LoadImageD,
    RandSpatialCropD,
    ScaleIntensityRangeD,
    SplitChannelD,
    ToTensorD,
)
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReservedKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.model import make_model_learnable, model_learnable_to_dxo
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.pt.pt_fed_utils import PTModelPersistenceFormatManager
from pt_constants import PTConstants
from simple_network import SimpleNetwork


class FLIP_TRAINER(Executor):
    def __init__(
        self,
        lr=0.01,
        epochs=5,
        train_task_name=AppConstants.TASK_TRAIN,
        submit_model_task_name=AppConstants.TASK_SUBMIT_MODEL,
        exclude_vars=None,
        project_id="",
        query="",
    ):
        """This CT Haemorrhage Trainer handles train and submit_model tasks. During train_task, it trains a
        3D Unet on paired CT images and segmentation labels. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            train_task_name (str, optional): Task name for train task. Defaults to "train".
            submit_model_task_name (str, optional): Task name for submit model. Defaults to "submit_model".
            exclude_vars (list): List of variables to exclude during model loading.
        """
        super(FLIP_TRAINER, self).__init__()

        self._lr = lr
        self._epochs = epochs
        self._train_task_name = train_task_name
        self._submit_model_task_name = submit_model_task_name
        self._exclude_vars = exclude_vars

        self.config = {}
        working_dir = Path(__file__).parent.resolve()
        with open(str(working_dir / "config.json")) as file:
            self.config = json.load(file)

        self.model = SimpleNetwork()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss = DiceLoss(include_background=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0, amsgrad=True)

        # Setup transforms using dictionary-based transforms.
        self._train_transforms = Compose(
            [
                LoadImageD(keys=["img", "seg"], reader="NiBabelReader", as_closest_canonical=False),
                AddChannelD(keys=["img", "seg"]),
                AddCoordinateChannelsD(keys=["img"], spatial_dims=(0, 1, 2)),
                # Rand3DElasticD(keys=['img', 'seg'], sigma_range=(1, 3), magnitude_range=(-10, 10), prob=0.5,
                #                mode=('bilinear', 'nearest'),
                #                rotate_range=(-0.34, 0.34),
                #                scale_range=(-0.1, 0.1), spatial_size=None),
                SplitChannelD(keys=["img"]),
                ScaleIntensityRangeD(keys=["img_0"], a_min=-15, a_max=100, b_min=-1, b_max=1, clip=True),
                ConcatItemsD(keys=["img_0", "img_1", "img_2", "img_3"], name="img"),
                DeleteItemsD(keys=["img_0", "img_1", "img_2", "img_3"]),
                RandSpatialCropD(keys=["img", "seg"], roi_size=(128, 128, 128), random_center=True, random_size=False),
                ToTensorD(keys=["img", "seg"]),
            ]
        )

        # Setup the training dataset
        self.flip = FLIP()
        self.project_id = project_id
        self.query = query
        self.dataframe = self.flip.get_dataframe(self.project_id, self.query)

        # Setup the persistence manager to save PT model.
        # The default training configuration is used by persistence manager
        # in case no initial model is found.
        self._default_train_conf = {"train": {"model": type(self.model).__name__}}
        self.persistence_manager = PTModelPersistenceFormatManager(
            data=self.model.state_dict(), default_train_conf=self._default_train_conf
        )

        self.project_id = project_id
        self.query = query

        train_dict = self.get_image_and_label_list(self.dataframe, val_split=self.config["VAL_SPLIT"])
        self._train_dataset = Dataset(train_dict, transform=self._train_transforms)
        self._train_loader = DataLoader(self._train_dataset, batch_size=1, shuffle=True, num_workers=1)
        self._n_iterations = len(self._train_loader)

    def get_image_and_label_list(self, dataframe, val_split=0.15):
        """Returns a list of dicts, each dict containing the path to an image and its corresponding label."""

        datalist = []
        # loop over each accession id in the train set
        for accession_id in dataframe["accession_id"]:
            try:
                image_data_folder_path = self.flip.get_by_accession_number(self.project_id, accession_id)
            except Exception as err:
                print(f"Could not get image data folder path for {accession_id}:")
                print(f"{err=}")
                print(f"{type(err)=}")
                print(f"{err.args=}")
                continue

            accession_folder_path = os.path.join(image_data_folder_path, accession_id)

            all_images = list(Path(accession_folder_path).rglob("sub-*_desc-affine_ct.nii"))

            this_accession_matches = 0
            print(f"Total base CT count found for accession_id {accession_id}: {len(all_images)}")
            for img in all_images:
                seg = str(img).replace("_ct", "_label-lesion_mask").replace(".nii", ".nii.gz")

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
                        f"Image dimensions ({img_header.shape}) do not match segmentation dimensions ({seg_header.shape})."
                    )
                    continue
                else:
                    datalist.append({"img": str(img), "seg": seg})
                    print(f"Matching base image and segmentation added.")
                    this_accession_matches += 1
            print(f"Added {this_accession_matches} matched image + segmentation pairs for {accession_id}.")
        print(f"Found {len(datalist)} files in train")

        # split into the training and testing data
        train_datalist, val_datalist = np.split(datalist, [int((1 - val_split) * len(datalist))])

        return train_datalist

    def local_train(self, fl_ctx, weights, abort_signal):
        # Set the model weights
        self.model.load_state_dict(state_dict=weights)

        # Basic training
        self.model.train()
        for epoch in range(self.config["LOCAL_ROUNDS"]):
            running_loss = 0.0
            num_images = 0
            for i, batch in enumerate(self._train_loader):

                if abort_signal.triggered:
                    # If abort_signal is triggered, we simply return.
                    # The outside function will check it again and decide steps to take.
                    return

                images, labels = batch["img"].to(self.device), batch["seg"].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                batch_size = images.shape[0]
                num_images += batch_size
                running_loss += cost.cpu().detach().numpy() * batch_size
                # print(f'Epoch: {epoch + 1}, Iteration: {i + 1}, Loss: {running_loss / num_images}')
            average_loss = running_loss / num_images
            self.log_info(
                fl_ctx,
                f"Epoch: {epoch}/{self.config['LOCAL_ROUNDS']}, Iteration: {i}, " f"Loss: {average_loss}",
            )

            self.flip.send_metrics_value(label="TRAIN_LOSS", value=average_loss, fl_ctx=fl_ctx)

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        if task_name == self._train_task_name:
            # Get model weights
            dxo = from_shareable(shareable)

            # Ensure data kind is weights.
            if not dxo.data_kind == DataKind.WEIGHTS:
                self.log_error(
                    fl_ctx,
                    f"data_kind expected WEIGHTS but got {dxo.data_kind} instead.",
                )
                return make_reply(ReturnCode.BAD_TASK_DATA)

            # Convert weights to tensor. Run training
            torch_weights = {k: torch.as_tensor(v) for k, v in dxo.data.items()}
            self.local_train(fl_ctx, torch_weights, abort_signal)

            # Check the abort_signal after training.
            # local_train returns early if abort_signal is triggered.
            if abort_signal.triggered:
                return make_reply(ReturnCode.TASK_ABORTED)

            # Save the local model after training.
            self.save_local_model(fl_ctx)

            # Get the new state dict and send as weights
            new_weights = self.model.state_dict()
            new_weights = {k: v.cpu().numpy() for k, v in new_weights.items()}

            outgoing_dxo = DXO(
                data_kind=DataKind.WEIGHTS,
                data=new_weights,
                meta={MetaKey.NUM_STEPS_CURRENT_ROUND: self._n_iterations},
            )
            return outgoing_dxo.to_shareable()
        elif task_name == self._submit_model_task_name:
            # Load local model
            ml = self.load_local_model(fl_ctx)

            # Get the model parameters and create dxo from it
            dxo = model_learnable_to_dxo(ml)
            return dxo.to_shareable()
        else:
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def save_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        ml = make_model_learnable(self.model.state_dict(), {})
        self.persistence_manager.update(ml)
        torch.save(self.persistence_manager.to_persistence_dict(), model_path)

    def load_local_model(self, fl_ctx: FLContext):
        run_dir = fl_ctx.get_engine().get_workspace().get_run_dir(fl_ctx.get_prop(ReservedKey.RUN_NUM))
        models_dir = os.path.join(run_dir, PTConstants.PTModelsDir)
        if not os.path.exists(models_dir):
            return None
        model_path = os.path.join(models_dir, PTConstants.PTLocalModelName)

        self.persistence_manager = PTModelPersistenceFormatManager(
            data=torch.load(model_path), default_train_conf=self._default_train_conf
        )
        ml = self.persistence_manager.to_model_learnable(exclude_vars=self._exclude_vars)
        return ml
