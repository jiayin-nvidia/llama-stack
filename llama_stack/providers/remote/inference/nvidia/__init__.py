# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

from pydantic import BaseModel, SecretStr

from llama_stack.apis.inference import Inference
from llama_stack.providers.remote.inference.nvidia.config import NVIDIAConfig
from llama_stack.providers.remote.inference.nvidia.nvidia import NVIDIAInferenceAdapter


class NVIDIAProviderDataValidator(BaseModel):
    """Provider data validator for NVIDIA inference."""

    nvidia_api_key: SecretStr | None = None


async def get_adapter_impl(config: NVIDIAConfig, _deps) -> Inference:
    """Get the NVIDIA inference adapter implementation."""
    if not isinstance(config, NVIDIAConfig):
        raise RuntimeError(f"Unexpected config type: {type(config)}")
    adapter = NVIDIAInferenceAdapter(config)
    return adapter


__all__ = ["NVIDIAConfig", "NVIDIAInferenceAdapter", "NVIDIAProviderDataValidator", "get_adapter_impl"]
