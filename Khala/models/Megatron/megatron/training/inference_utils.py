# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Small Megatron compatibility surface required by Khala inference."""

import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.transformer.module import Float16Module
from megatron.core.utils import get_model_config

from .global_vars import get_args
from .utils import to_empty_if_meta_device


def get_model(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
    """Build a model for inference without importing Megatron's pretraining loop."""
    if wrap_with_ddp:
        raise RuntimeError("Khala's inference-only build does not support DDP wrapping.")

    args = get_args()
    args.model_type = model_type

    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            model = []
            for vp_stage in range(args.virtual_pipeline_model_parallel_size):
                pre_process = mpu.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
                post_process = mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
                this_model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    vp_stage=vp_stage,
                )
                this_model.model_type = model_type
                this_model.vp_stage = vp_stage
                model.append(this_model)
            return model

        model = model_provider_func(
            pre_process=mpu.is_pipeline_first_stage(),
            post_process=mpu.is_pipeline_last_stage(),
        )
        model.model_type = model_type
        return model

    if args.init_model_with_meta_device:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]

    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    num_parameters = sum(
        sum(param.nelement() for param in model_module.parameters()) for model_module in model
    )
    if mpu.get_data_parallel_rank() == 0 and mpu.get_context_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank "
            f"({mpu.get_tensor_model_parallel_rank()}, "
            f"{mpu.get_pipeline_model_parallel_rank()}): {num_parameters}",
            flush=True,
        )

    if not args.init_model_with_meta_device:
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    if args.init_model_with_meta_device:
        model = [to_empty_if_meta_device(model_module, device=torch.device("cuda")) for model_module in model]

    correct_amax_history_if_needed(model)
    return model


def pretrain(*_args, **_kwargs):
    raise RuntimeError("Megatron pretraining is not included in this inference-only Khala build.")


def get_train_valid_test_num_samples():
    raise RuntimeError("Dataset sizing is not included in this inference-only Khala build.")
