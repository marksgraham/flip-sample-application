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

import torch
from torch import nn
import torch.cuda
from monai.networks.nets import BasicUNet


class SimpleNetwork(nn.Module):
    """
    Wraps a MONAI BasicUNet allowing the choice of returning the logits or sigmoided logits. This is useful
    because we train on patches, but evaluate on full images using a sliding window approach. We need to return
    logits for the sliding window approach, but sigmoided logits for the patch training approach.
    """

    def __init__(self, num_classes=1):
        super().__init__()
        self.net = BasicUNet(dimensions=3, features=(32, 32, 64, 128, 256, 32), in_channels=4, out_channels=num_classes)

    def forward(self, x, do_sigmoid=True):
        logits = self.net(x)
        if do_sigmoid:
            return torch.sigmoid(logits)
        else:
            return logits
