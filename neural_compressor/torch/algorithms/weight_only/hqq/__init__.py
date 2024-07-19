# Copyright (c) 2024 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .quantizer import HQQuantizer
from .config import HQQModuleConfig, QTensorConfig

class HQQuantizer:
    """
    A class for quantizing models using the HQQ algorithm.

    Attributes:
        quant_config (ConfigMappingType): Configuration for quantization.

    Methods:
        prepare(model: torch.nn.Module, *args, **kwargs) -> Optional[torch.nn.Module]:
            Prepares a given model for quantization.
        convert(model: torch.nn.Module, *args, **kwargs) -> Optional[torch.nn.Module]:
            Converts a prepared model to a quantized model.
        save(model, path):
            Saves the quantized model to the specified path.
    """

class HQQModuleConfig:
    """
    Configuration for HQQ modules.

    Attributes:
        weight (QTensorConfig): Configuration for weight quantization.
        scale (QTensorConfig): Configuration for scale quantization.
        zero (QTensorConfig): Configuration for zero quantization.

    Methods:
        __repr__() -> str:
            Returns a string representation of the HQQModuleConfig object.
    """

class QTensorConfig:
    """
    Configuration for quantized tensors.

    Attributes:
        nbits (int): Number of bits for quantization.
        channel_wise (bool): Whether to use channel-wise quantization.
        group_size (int): Size of the quantization group.
        optimize (bool): Whether to optimize the quantization.
        round_zero (Optional[bool]): Whether to round zero.
        pack (bool): Whether to pack the quantized tensor.

    Methods:
        __repr__() -> str:
            Returns a string representation of the QTensorConfig object.
    """
