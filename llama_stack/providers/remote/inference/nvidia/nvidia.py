# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import logging

from llama_stack.apis.inference import (
    ChatCompletionRequest,
)
from llama_stack.providers.utils.inference.litellm_openai_mixin import LiteLLMOpenAIMixin

from . import NVIDIAConfig
from .models import MODEL_ENTRIES
from .utils import _is_nvidia_hosted

logger = logging.getLogger(__name__)


class NVIDIAInferenceAdapter(LiteLLMOpenAIMixin):
    def __init__(self, config: NVIDIAConfig) -> None:
        if _is_nvidia_hosted(config):
            if not config.api_key:
                raise RuntimeError(
                    "API key is required for hosted NVIDIA NIM. Either provide an API key or use a self-hosted NIM."
                )

        base_url = f"{config.url}/v1" if config.append_api_version else config.url

        # Special model URLs for vision models
        special_model_urls = {
            "meta/llama-3.2-11b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct",
            "meta/llama-3.2-90b-vision-instruct": "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-90b-vision-instruct",
        }

        # Initialize LiteLLMOpenAIMixin
        LiteLLMOpenAIMixin.__init__(
            self,
            model_entries=MODEL_ENTRIES,
            api_key_from_config=config.api_key.get_secret_value() if config.api_key else None,
            provider_data_api_key_field="nvidia_api_key",
            openai_compat_api_base=base_url,
        )

        self._config = config
        self._special_model_urls = special_model_urls

    async def _get_params(self, request: ChatCompletionRequest) -> dict:
        """Override _get_params to handle NVIDIA-specific parameters"""
        params = await super()._get_params(request)

        # Handle special model URLs
        model_id = request.model
        if _is_nvidia_hosted(self._config) and model_id in self._special_model_urls:
            params["api_base"] = self._special_model_urls[model_id]

        # Add NVIDIA-specific parameters
        nvext = {}

        # Handle sampling parameters
        if request.sampling_params:
            nvext["repetition_penalty"] = request.sampling_params.repetition_penalty

            strategy = request.sampling_params.strategy
            if hasattr(strategy, "top_k"):
                if strategy.top_k != -1 and strategy.top_k < 1:
                    logger.warning("top_k must be -1 or >= 1")
                nvext["top_k"] = strategy.top_k

        # Add NVIDIA extensions if any are set
        if nvext:
            params["extra_body"] = {"nvext": nvext}

        return params

    def get_api_key(self) -> str:
        """Get the API key from provider data or config."""
        try:
            provider_data = self.get_request_provider_data()
            key_field = self.provider_data_api_key_field
            if provider_data and getattr(provider_data, key_field, None):
                api_key = getattr(provider_data, key_field)
            else:
                api_key = self.api_key_from_config
        except ValueError:
            # No provider data validator available, use config
            api_key = self.api_key_from_config

        if not api_key and _is_nvidia_hosted(self._config):
            raise RuntimeError(
                "API key is required for hosted NVIDIA NIM. Either provide an API key or use a self-hosted NIM."
            )

        return api_key or "NO KEY"
