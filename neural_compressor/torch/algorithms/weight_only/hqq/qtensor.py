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

from dataclasses import asdict, dataclass
from typing import Tuple, Union

import torch

__all__ = [
    "QTensor",
    "QTensorMetaInfo",
]


@dataclass
class QTensorMetaInfo:
    """
    Meta information for quantized tensors.

    Attributes:
        nbits (int): Number of bits for quantization.
        group_size (int): Size of the quantization group.
        shape (Tuple): Shape of the tensor.
        axis (int): Axis for quantization.
        packing (bool): Whether the tensor is packed.
    """
    nbits: int
    group_size: int
    shape: Tuple
    axis: int
    packing: bool

    def to_dict(self):
        """
        Converts the QTensorMetaInfo object to a dictionary.

        Returns:
            dict: A dictionary representation of the QTensorMetaInfo object.
        """
        return asdict(self)


class QTensor:
    """
    A class representing a quantized tensor.

    Attributes:
        val (torch.Tensor): The quantized tensor values.
        scale (Union[torch.Tensor, "QTensor"], optional): The scale tensor or quantized scale tensor.
        zero (Union[torch.Tensor, "QTensor"], optional): The zero tensor or quantized zero tensor.
        meta_info (QTensorMetaInfo, optional): Meta information for the quantized tensor.

    Methods:
        is_scale_quantized() -> bool:
            Checks if the scale is quantized.
        is_zero_quantized() -> bool:
            Checks if the zero is quantized.
        _get_scale_repr() -> str:
            Returns a string representation of the scale.
        _get_zero_repr() -> str:
            Returns a string representation of the zero.
        __repr__() -> str:
            Returns a string representation of the QTensor object.
        to(*args, **kwargs):
            Moves the tensor to the specified device and dtype.
        half():
            Converts the tensor to half precision.
        to_state_dict() -> dict:
            Converts the QTensor object to a state dictionary.
    """
    val: torch.Tensor
    scale: Union[torch.Tensor, "QTensor"] = None
    zero: Union[torch.Tensor, "QTensor"] = None
    meta_info: QTensorMetaInfo = None
    """
    val: torch.Tensor
    scale:
        val: torch.Tensor
        scale: torch.Tensor
        zero: torch.Tensor
    zero:
        torch.Tensor
    """

    def __init__(self, val, scale=None, zero=None, meta_info=None):
        self.val = val
        self.scale = scale
        self.zero = zero
        self.meta_info = meta_info

    def is_scale_quantized(self) -> bool:
        """
        Checks if the scale is quantized.

        Returns:
            bool: True if the scale is quantized, False otherwise.
        """
        return isinstance(self.scale, QTensor)

    def is_zero_quantized(self) -> bool:
        """
        Checks if the zero is quantized.

        Returns:
            bool: True if the zero is quantized, False otherwise.
        """
        return isinstance(self.zero, QTensor)

    def _get_scale_repr(self) -> str:
        """
        Returns a string representation of the scale.

        Returns:
            str: A string representation of the scale.
        """
        if not self.is_scale_quantized():
            if self.scale is not None:
                return (
                    f"scale_shape={self.scale.shape}, "
                    f"scale_dtype={self.scale.dtype}, "
                    f"scale_device={self.scale.device}\n"
                )
            else:
                return "scale is None\n"
        else:
            return self.scale.__repr__() + "\n"

    def _get_zero_repr(self) -> str:
        """
        Returns a string representation of the zero.

        Returns:
            str: A string representation of the zero.
        """
        if not self.is_zero_quantized():
            if self.zero is not None:
                return (
                    f"zero_shape={self.zero.shape}, "
                    f"zero_dtype={self.zero.dtype}, "
                    f"zero_device={self.zero.device}\n"
                )
            else:
                return "zero is None\n"
        else:
            return self.zero.__repr__() + "\n"

    def __repr__(self) -> str:
        """
        Returns a string representation of the QTensor object.

        Returns:
            str: A string representation of the QTensor object.
        """
        # TODO: refine it later
        return (
            f"QTensor(\n"
            f"val_shape={self.val.shape}, val_dtype={self.val.dtype}, val_device={self.val.device}\n"
            f"scale_quantized={self.is_scale_quantized()},\n"
            f"zero_quantized={self.is_zero_quantized()},\n"
            f"zero=({self._get_zero_repr()})"
            f"scale=({self._get_scale_repr()})"
            f"meta_info={self.meta_info}\n)"
        )

    def to(self, *args, **kwargs):
        """
        Moves the tensor to the specified device and dtype.

        Args:
            *args: Positional arguments for the `to` method.
            **kwargs: Keyword arguments for the `to` method.

        Returns:
            QTensor: The QTensor object moved to the specified device and dtype.
        """
        self.val = self.val.to(*args, **kwargs)
        self.scale = self.scale.to(*args, **kwargs)
        self.zero = self.zero.to(*args, **kwargs)
        return self

    def half(self):
        """
        Converts the tensor to half precision.

        Returns:
            QTensor: The QTensor object in half precision.
        """
        # TODO: refine it later
        if self.val.dtype == torch.float32:
            self.val = self.val.half()
        if self.scale is not None:
            self.scale = self.scale.half()
        if self.zero is not None:
            self.zero = self.zero.half()
        return self

    def to_state_dict(self):
        """
        Converts the QTensor object to a state dictionary.

        Returns:
            dict: A state dictionary representation of the QTensor object.
        """
        state = {}
        state["val"] = self.val
        state["meta_info"] = self.meta_info.to_dict()
        state["scale_quantized"] = self.is_scale_quantized()
        state["zero_quantized"] = self.is_zero_quantized()
        if self.is_scale_quantized():
            state["meta_info"]["scale"] = self.scale.to_state_dict()
        else:
            state["meta_info"]["scale"] = self.scale
        if self.is_zero_quantized():
            state["meta_info"]["zero"] = self.zero.to_state_dict()
        else:
            state["meta_info"]["zero"] = self.zero
        return state
