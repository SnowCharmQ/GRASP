# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import json
import os
import random
import time
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import trl
from accelerate.utils import gather_object, is_peft_model
from packaging import version
from transformers import PreTrainedModel
from trl import GKDTrainer as HFGKDTrainer
from trl import SFTTrainer as HFSFTTrainer

from swift.llm.template.template_inputs import TemplateInputs
from swift.utils import (JsonlWriter, get_logger, is_swanlab_available, is_wandb_available, remove_response,
                         unwrap_model_for_generation)
from ..mixin import SwiftMixin
from .rollout_mixin import DataType, RolloutTrainerMixin
from .utils import (get_gather_if_zero3_context, identity_data_collator, patch_profiling_context,
                    patch_profiling_decorator, prepare_deepspeed)

# Import GRPO methods for reuse
try:
    from .grpo_trainer import GRPOTrainer
    _GRPO_AVAILABLE = True
except ImportError:
    _GRPO_AVAILABLE = False
    GRPOTrainer = None

try:
    from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
    _liger_kernel_available = True
except ImportError:
    _liger_kernel_available = False

del HFGKDTrainer.__init__
del HFSFTTrainer.__init__

logger = get_logger()
if is_wandb_available():
    import wandb
if is_swanlab_available():
    import swanlab


class DataSource(str, Enum):
    STUDENT = 'student'  # On-policy: student model generates responses
    TEACHER = 'teacher'  # Sequential KD: teacher model generates responses
    DATASET = 'dataset'  # Off-policy: use dataset responses


class GKDTrainer(RolloutTrainerMixin, SwiftMixin, HFGKDTrainer):

    def __init__(self, model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, *_args, **kwargs):
        teacher_model = kwargs.pop('teacher_model')
        teacher_deepspeed_config = kwargs.pop('teacher_deepspeed_config', None)
        self.vllm_client = kwargs.pop('vllm_client', None)
        kwargs['data_collator'] = identity_data_collator
        super().__init__(model, None, *_args, **kwargs)
        # Get args from kwargs (passed to __init__) or from self.args (set by parent class)
        args_from_kwargs = kwargs.get('args', None)
        args_from_self = getattr(self, 'args', None)
        
        # Prefer args from kwargs, fallback to self.args
        args = args_from_kwargs if args_from_kwargs is not None else args_from_self
        
        if args is None:
            logger.warning("[DEBUG __init__] args is None, cannot initialize GKD-specific parameters")
            self.use_on_policy_distillation = False
        else:
            self.lmbda = args.lmbda
            self.temperature = args.temperature
            self.seq_kd = args.seq_kd
            # Store use_on_policy_distillation flag explicitly
            # Try to get from args object first, then from self.args
            use_on_policy_distillation_value = getattr(args, 'use_on_policy_distillation', None)
            if use_on_policy_distillation_value is None and args_from_self is not None:
                use_on_policy_distillation_value = getattr(args_from_self, 'use_on_policy_distillation', False)
            self.use_on_policy_distillation = use_on_policy_distillation_value if use_on_policy_distillation_value is not None else False
            logger.info(f"[DEBUG __init__] args_from_kwargs type={type(args_from_kwargs)}, args_from_self type={type(args_from_self)}, use_on_policy_distillation={self.use_on_policy_distillation}, args.use_on_policy_distillation={getattr(args, 'use_on_policy_distillation', 'NOT_FOUND') if args else 'NO_ARGS'}")
        self.generation_config = model.generation_config
        self._metrics = {'train': defaultdict(list), 'eval': defaultdict(list)}
        self._total_train_tokens = 0
        
        # Initialize attributes reused from GRPO pattern
        self.is_multimodal = model.model_meta.is_multimodal
        self.model_kwarg_keys = (
            inspect.signature(model.forward).parameters.keys() if not hasattr(model, 'get_base_model') else
            inspect.signature(model.get_base_model().forward).parameters.keys())
        
        # Initialize metrics for on-policy distillation
        for mode in ['train', 'eval']:
            self._metrics[mode]['reward'] = []  # reverse_kl as reward
            self._metrics[mode]['reverse_kl'] = []
            self._metrics[mode]['advantage'] = []
            self._metrics[mode]['importance_weight'] = []

        # Initialize logging components
        self._prepare_logging()

        # Initialize liger loss
        self._prepare_liger_loss()

        self.teacher_ds3_gather_for_generation = args.ds3_gather_for_generation
        self.is_teacher_ds3 = None
        # Initialize teacher model
        if self.is_deepspeed_enabled:
            if teacher_deepspeed_config is not None:
                self.is_teacher_ds3 = teacher_deepspeed_config.get('zero_optimization', {}).get('stage') == 3
                if not self.is_teacher_ds3:
                    self.teacher_ds3_gather_for_generation = False
                self.teacher_model = prepare_deepspeed(
                    teacher_model, self.accelerator, deepspeed_config=teacher_deepspeed_config, training_args=args)
            else:
                self.teacher_model = prepare_deepspeed(teacher_model, self.accelerator)
        elif self.is_fsdp_enabled:
            from .utils import prepare_fsdp
            self.teacher_model = prepare_fsdp(teacher_model, self.accelerator)
        else:
            self.teacher_model = self.accelerator.prepare_model(teacher_model, evaluation_mode=True)
        self.teacher_model.eval()
        if self.args.offload_teacher_model:
            self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

        # Initialize rollout infrastructure for vLLM support
        if args.use_vllm:
            self.prepare_rollout()
            logger.info('vLLM engine initialized for GKD training')
        else:
            # Initialize max_completion_length even when not using vLLM
            # This is needed for generate_on_policy_outputs to set max_new_tokens correctly
            if hasattr(args, 'max_completion_length'):
                self.max_completion_length = args.max_completion_length
            else:
                # Fallback: use generation_config's max_new_tokens if available
                self.max_completion_length = getattr(self.generation_config, 'max_new_tokens', None)
                if self.max_completion_length is None:
                    logger.warning("[DEBUG __init__] max_completion_length not found in args and generation_config, using default 512")
                    self.max_completion_length = 512
            logger.info(f"[DEBUG __init__] Initialized max_completion_length={self.max_completion_length} (not using vLLM)")

        # Initialize activation offloading context
        args.activation_offloading = False  # TODO: remove
        if args.activation_offloading:
            from trl.models import get_act_offloading_ctx_manager
            self.maybe_activation_offload_context = get_act_offloading_ctx_manager(model=self.model)
        else:
            self.maybe_activation_offload_context = nullcontext()
        self._trl_version_gte_0_24 = version.parse(trl.__version__) >= version.parse('0.24')

        # Initialize resample data iterator for truncation_strategy 'raise'('delete')
        if self.template.truncation_strategy == 'raise':
            self._prepare_resample_data_iterator()

        # On-Policy Distillation: accumulate teacher forward time; wall time saved in train() finally
        self._opd_teacher_forward_seconds_local = 0.0

    @contextmanager
    def _opd_teacher_forward_timer(self):
        """Accumulate wall time for teacher forward only; does not alter model execution."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._opd_teacher_forward_seconds_local += time.perf_counter() - t0

    def _save_opd_timing_stats(self, wall_seconds: float) -> None:
        """Save OPD teacher time and total training wall time to output_dir (main process only)."""
        if not getattr(self, 'use_on_policy_distillation', False):
            return
        local_teacher = float(self._opd_teacher_forward_seconds_local)
        global_teacher_sum = local_teacher
        if torch.distributed.is_initialized():
            t = torch.tensor([local_teacher], device=self.args.device, dtype=torch.float64)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
            global_teacher_sum = float(t.item())
        if not self.accelerator.is_main_process:
            return
        payload = {
            'training_wall_seconds': float(wall_seconds),
            'teacher_forward_seconds_this_rank': local_teacher,
            'teacher_forward_seconds_sum_all_ranks': global_teacher_sum,
            'world_size': int(self.accelerator.num_processes),
        }
        path = os.path.join(self.args.output_dir, 'opd_timing_stats.json')
        os.makedirs(self.args.output_dir, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f'[OPD] Saved timing stats to {path}: {payload}')

    def train(self, *args, **kwargs):
        """Record total wall-clock training time; on exit save OPD teacher vs total time if OPD enabled."""
        t_wall = time.perf_counter()
        try:
            return super().train(*args, **kwargs)
        finally:
            wall = time.perf_counter() - t_wall
            try:
                self._save_opd_timing_stats(wall_seconds=wall)
            except Exception as e:
                logger.warning(f'[OPD] Failed to save opd_timing_stats.json: {e}')

    # Code borrowed from huggingface/trl
    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs using the model.

        When encode_prompt_only=True, inputs['input_ids'] already contains only the prompt part.
        """
        assert not self.template.padding_free, 'generate not support padding_free/packing.'
        prompt_input_ids = inputs['input_ids']
        model_inputs = {k: v for k, v in inputs.items() if k != 'labels'}
        model_inputs.pop('position_ids', None)
        model_inputs.pop('text_position_ids', None)
        kwargs = {}
        base_model = self.template.get_base_model(model)
        parameters = inspect.signature(base_model.generate).parameters
        if 'use_model_defaults' in parameters:
            kwargs['use_model_defaults'] = False
        
        # Set max_new_tokens from max_completion_length if available
        # This ensures we generate the desired length instead of using model's default
        if hasattr(self, 'max_completion_length') and self.max_completion_length is not None:
            # Create a copy of generation_config to avoid modifying the original
            from copy import deepcopy
            generation_config = deepcopy(generation_config)
            generation_config.max_new_tokens = self.max_completion_length
            logger.info(f"[DEBUG generate_on_policy_outputs] Setting max_new_tokens={self.max_completion_length} from max_completion_length")
        
        with self.template.generate_context():
            if self.model.model_meta.is_multimodal:
                _, model_inputs = self.template.pre_forward_hook(model, None, model_inputs)
            generated_outputs = model.generate(
                **model_inputs, generation_config=generation_config, return_dict_in_generate=True, **kwargs)
        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        if not self.template.skip_prompt:
            generated_tokens = torch.concat([prompt_input_ids, generated_tokens], dim=1)
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()
        new_labels[:, :prompt_input_ids.shape[1]] = -100

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        new_position_ids = new_attention_mask.cumsum(dim=1) - 1
        new_position_ids[new_position_ids < 0] = 0
        inputs['position_ids'] = new_position_ids
        return generated_tokens, new_attention_mask, new_labels

    @patch_profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get data source: DataSource.STUDENT, DataSource.TEACHER, or DataSource.DATASET
        # Use get() instead of pop() to avoid removing the field in case _prepare_inputs is called again
        data_source = inputs.get('_data_source', DataSource.DATASET)
        # Try multiple ways to get the flag - prefer stored attribute, then args
        use_on_policy_distillation_flag = getattr(self, 'use_on_policy_distillation', None)
        if use_on_policy_distillation_flag is None:
            use_on_policy_distillation_flag = getattr(self.args, 'use_on_policy_distillation', False)
        if isinstance(use_on_policy_distillation_flag, str):
            use_on_policy_distillation_flag = use_on_policy_distillation_flag.lower() in ('true', '1', 'yes')
        has_old_logps = inputs.get('old_per_token_logps') is not None
        
        # Check if we should use on-policy distillation (RL-based) for student-generated samples
        use_on_policy_distillation = (
            data_source == DataSource.STUDENT and 
            use_on_policy_distillation_flag and
            has_old_logps
        )
        
        # Prepare model inputs (same as original code)
        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        # If generate is used, then use_logits_to_keep must be set to False.
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep and not self.use_liger_gkd_loss:
            self.prepare_logits_to_keep(inputs)
            model_inputs['logits_to_keep'] = inputs['logits_to_keep']

        if self.use_liger_gkd_loss:
            # Liger fused JSD loss for memory efficiency
            # Get base models (exclude lm_head to save memory)
            unwrapped_student = self.accelerator.unwrap_model(model)
            if is_peft_model(unwrapped_student):
                unwrapped_student = unwrapped_student.base_model.model
            base_student = getattr(unwrapped_student, getattr(unwrapped_student, 'base_model_prefix', 'model'),
                                   unwrapped_student)

            unwrapped_teacher = self.accelerator.unwrap_model(self.teacher_model)
            base_teacher = getattr(unwrapped_teacher, getattr(unwrapped_teacher, 'base_model_prefix', 'model'),
                                   unwrapped_teacher)

            # Forward through base models
            student_outputs = base_student(**model_inputs, use_cache=False)

            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with load_context:
                with torch.no_grad():
                    teacher_outputs = base_teacher(**model_inputs, use_cache=False)

                # Get hidden states (shifted)
                student_hidden = student_outputs.last_hidden_state[:, :-1]
                teacher_hidden = teacher_outputs.last_hidden_state[:, :-1]

                # Release full outputs to free memory
                del student_outputs, teacher_outputs

                # Prepare labels (shifted)
                labels_mask = inputs['labels'] != -100
                masked_input_ids = torch.where(labels_mask, inputs['input_ids'],
                                               torch.full_like(inputs['input_ids'], -100))
                true_labels = masked_input_ids[:, 1:].contiguous()

                # Release intermediate tensors
                del labels_mask, masked_input_ids

                # Get output heads
                student_head = unwrapped_student.get_output_embeddings()
                teacher_head = unwrapped_teacher.get_output_embeddings()

                # Prepare context managers for gathering parameters in zero3
                teacher_context = get_gather_if_zero3_context(self, is_zero3=self.is_teacher_ds3)(teacher_head.weight)
                student_context = get_gather_if_zero3_context(self)(student_head.weight)

                with teacher_context, student_context:
                    # Compute liger fused JSD loss
                    loss = self.liger_jsd_loss(
                        student_input=student_hidden,
                        student_weight=student_head.weight,
                        teacher_input=teacher_hidden,
                        teacher_weight=teacher_head.weight,
                        true_labels=true_labels,
                        student_bias=getattr(student_head, 'bias', None),
                        teacher_bias=getattr(teacher_head, 'bias', None),
                    )
                    # loss / grad norm is unexpectedly large, normalize by sequence length
                    # https://github.com/linkedin/Liger-Kernel/blob/v0.6.3/src/liger_kernel/chunked_loss/jsd_loss.py#L9-L39
                    loss /= student_hidden.shape[1]
                # Release hidden states after loss computation
                del student_hidden, teacher_hidden, true_labels
        else:
            # Standard loss computation (same as original code)
            if self.args.sft_alpha > 0:
                model_inputs['labels'] = inputs['labels']
            # compute student output
            outputs_student = model(**model_inputs)

            model_inputs.pop('labels', None)
            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            
            # Check if we should use OPD loss instead of standard JSD loss
            if use_on_policy_distillation:
                # Use OPD loss: teacher should compute logprobs for student-generated trajectory
                # Teacher should use the same method as _get_per_token_logps_and_entropies
                # to ensure it only computes logprobs for the completion part (student-generated trajectory)
                old_per_token_logps = inputs['old_per_token_logps']
                # Determine logits_to_keep from old_per_token_logps (sampling logprobs)
                if old_per_token_logps.dim() == 2:
                    logits_to_keep_for_teacher = old_per_token_logps.shape[1]
                else:
                    labels = inputs['labels']
                    logits_to_keep_for_teacher = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
                
                # Prepare inputs for teacher logprobs computation
                # Teacher should compute logprobs for the student-generated trajectory (full input_ids)
                # The input_ids in inputs already contains prompt + student-generated completion
                teacher_inputs_for_logprobs = inputs.copy()
                teacher_inputs_for_logprobs['logits_to_keep'] = logits_to_keep_for_teacher
                teacher_inputs_for_logprobs.pop('old_per_token_logps', None)
                teacher_inputs_for_logprobs.pop('_data_source', None)
                
                # Compute teacher logprobs using the same method as student
                # This ensures teacher only computes logprobs for completion tokens (student-generated trajectory)
                # Teacher computes logprobs for the full trajectory (prompt + student completion),
                # but only returns logprobs for completion tokens (via logits_to_keep)
                with torch.no_grad(), load_context:
                    with self._opd_teacher_forward_timer():
                        teacher_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.teacher_model, teacher_inputs_for_logprobs, compute_entropy=False)
                inputs['teacher_per_token_logps'] = teacher_per_token_logps
                # logger.info(f"[DEBUG compute_loss] Computed teacher_per_token_logps for student-generated trajectory, shape={teacher_per_token_logps.shape}, logits_to_keep={logits_to_keep_for_teacher}")
                # Use OPD loss computation
                return self._compute_on_policy_distillation_loss(model, inputs, return_outputs, num_items_in_batch)
            
            # Standard JSD loss: compute teacher outputs for standard loss
            with torch.no_grad(), load_context:
                outputs_teacher = self.teacher_model(**model_inputs)
            
            # Standard JSD loss computation (original code)
            shifted_labels = torch.roll(inputs['labels'], shifts=-1, dims=1)
            mask = shifted_labels != -100
            shifted_student_logits = outputs_student.logits[mask][None]
            shifted_teacher_logits = outputs_teacher.logits[mask][None]

            # Fix the vocab_size mismatch between Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct.
            stu_dim = shifted_student_logits.shape[-1]
            tea_dim = shifted_teacher_logits.shape[-1]
            if stu_dim < tea_dim:
                shifted_student_logits = F.pad(shifted_student_logits, (0, tea_dim - stu_dim), 'constant', 0)
                shifted_student_logits[..., stu_dim:] = shifted_teacher_logits[..., stu_dim:]
            elif stu_dim > tea_dim:
                shifted_teacher_logits = F.pad(shifted_teacher_logits, (0, stu_dim - tea_dim), 'constant', 0)
                shifted_teacher_logits[..., tea_dim:] = shifted_student_logits[..., tea_dim:]

            # compute loss
            loss = self.generalized_jsd_loss(
                student_logits=shifted_student_logits,
                teacher_logits=shifted_teacher_logits,
                beta=self.beta,
            )
            if self._trl_version_gte_0_24:
                loss /= shifted_student_logits.shape[1]
            # Add SFT loss if enabled (skip for student-generated responses)
            if self.args.sft_alpha > 0 and data_source != DataSource.STUDENT:
                loss = loss + self.args.sft_alpha * outputs_student.loss

        # Return loss
        if return_outputs:
            if self.use_liger_gkd_loss:
                # outputs has been released in liger loss computation to reduce peak memory
                outputs_student = None
            return (loss, outputs_student)
        else:
            return loss

    def _prepare_batch_inputs(self, inputs: list, encode_prompt_only: bool = False) -> Dict[str, torch.Tensor]:
        """Prepare batch inputs for training.

        Args:
            inputs: List of input data dictionaries
            encode_prompt_only: If True, only encode the prompt part (for on-policy/seq_kd generation).
                               If False, encode the full messages including response (for offline dataset).
        """
        from swift.llm import to_device
        from .utils import replace_assistant_response_with_ids

        template = self.template
        batch_encoded_inputs = []

        # Use 'pt' mode for prompt-only encoding, 'train' mode for full encoding
        mode = 'pt' if encode_prompt_only else 'train'
        with self._template_context(template, mode=mode):
            for data in inputs:
                if 'response_token_ids' in data and data['response_token_ids']:
                    data['messages'] = replace_assistant_response_with_ids(data['messages'], data['response_token_ids'])

                if encode_prompt_only:
                    # Remove response content for prompt-only encoding
                    messages = data.get('messages', [])
                    if messages and messages[-1].get('role') == 'assistant':
                        messages[-1]['content'] = None

                encoded = template.encode(data, return_length=True)
                batch_encoded_inputs.append(encoded)

            batch_encoded = to_device(template.data_collator(batch_encoded_inputs), self.model.device)

        return batch_encoded

    # Code borrowed from huggingface/trl
    @patch_profiling_decorator
    def training_step(self,
                      model: nn.Module,
                      inputs: DataType,
                      num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        """
        Perform a training step for the Generalized Knowledge Distillation (GKD) model.

        This method implements the on-policy learning approach described in the GKD paper.
        With probability `self.lmbda`, it generates new responses using the student model,
        which are then used for training instead of the original inputs.

        When use_vllm is enabled, vLLM engine is used for faster generation.
        """
        args = self.args
        with patch_profiling_context(self, 'get_completions'):
            if self._get_random_num() <= self.lmbda:
                # On-policy: student model generates responses
                data_source = DataSource.STUDENT
                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                if args.use_vllm:
                    processed_inputs = self._preprocess_inputs(inputs)
                    generated_inputs = self._fast_infer(processed_inputs)
                    if self.log_completions:
                        messages = [inp['messages'][:-1] for inp in generated_inputs]
                        completions = [deepcopy(inp['messages'][-1]['content']) for inp in generated_inputs]
                        valid_messages = gather_object(messages)
                        valid_completions = gather_object(completions)
                        self._logs['prompt'].extend(self._apply_chat_template_to_messages_list(valid_messages))
                        self._logs['completion'].extend(valid_completions)
                    with self._template_context(self.template):
                        # vLLM already generated response, encode full messages
                        encoded_inputs = self._prepare_batch_inputs(generated_inputs, encode_prompt_only=False)
                    
                    # Save logprobs from vLLM rollout for on-policy distillation
                    if getattr(args, 'use_on_policy_distillation', False):
                        encoded_inputs['old_per_token_logps'] = self._get_vllm_sampling_logprobs(
                            model, encoded_inputs)
                else:
                    # Need prompt-only encoding for on-policy generation
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                    with unwrap_model_for_generation(
                            model, self.accelerator,
                            gather_deepspeed3_params=args.ds3_gather_for_generation) as unwrapped_model:
                        unwrapped_model.eval()
                        new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                            unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                        unwrapped_model.train()
                    # override with generated inputs
                    encoded_inputs['input_ids'] = new_input_ids
                    encoded_inputs['attention_mask'] = new_attention_mask
                    encoded_inputs['labels'] = new_labels
                    
                    # Save logprobs from sampling time for on-policy distillation
                    # Important: Compute logprobs in eval mode to match training-time computation
                    if getattr(args, 'use_on_policy_distillation', False):
                        logger.info("[DEBUG training_step] Computing old_per_token_logps for on-policy distillation")
                        # Set model to eval mode and disable gradients for logprob computation
                        was_training = unwrapped_model.training
                        unwrapped_model.eval()
                        with torch.no_grad():
                            encoded_inputs['old_per_token_logps'] = self._get_sampling_logprobs(
                                unwrapped_model, encoded_inputs)
                        # Restore training mode if it was training before
                        if was_training:
                            unwrapped_model.train()
                        logger.info(f"[DEBUG training_step] old_per_token_logps shape: {encoded_inputs['old_per_token_logps'].shape if encoded_inputs.get('old_per_token_logps') is not None else None}")

            elif self.seq_kd:
                # Sequential KD: teacher model generates responses
                data_source = DataSource.TEACHER

                # Resample inputs that fail encoding when truncation_strategy is 'raise'('delete')
                if self.template.truncation_strategy == 'raise':
                    inputs = self.resample_encode_failed_inputs(inputs)
                # Need prompt-only encoding for teacher generation
                encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=True)
                load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
                with load_context, unwrap_model_for_generation(
                        self.teacher_model,
                        self.accelerator,
                        gather_deepspeed3_params=self.teacher_ds3_gather_for_generation) as unwrapped_model:
                    unwrapped_model.eval()
                    new_input_ids, new_attention_mask, new_labels = self.generate_on_policy_outputs(
                        unwrapped_model, encoded_inputs, self.generation_config, self.processing_class.pad_token_id)
                # override with generated inputs
                encoded_inputs['input_ids'] = new_input_ids
                encoded_inputs['attention_mask'] = new_attention_mask
                encoded_inputs['labels'] = new_labels

                # Save teacher logprobs from sampling time for on-policy distillation
                # This avoids recomputing teacher logprobs in _compute_on_policy_distillation_loss
                if getattr(args, 'use_on_policy_distillation', False):
                    logger.info("[DEBUG training_step] Computing teacher_per_token_logps from teacher sampling")
                    with torch.no_grad():
                        # Prepare inputs for logprob computation
                        teacher_inputs_for_logprobs = encoded_inputs.copy()
                        # Use the same method as _compute_on_policy_distillation_loss
                        if 'logits_to_keep' not in teacher_inputs_for_logprobs:
                            labels = teacher_inputs_for_logprobs.get('labels')
                            if labels is not None:
                                completion_mask = (labels != -100)
                                if completion_mask.any():
                                    if labels.shape[0] == 1:
                                        logits_to_keep = completion_mask.sum().item()
                                    else:
                                        logits_to_keep = completion_mask.sum(dim=-1).max().item()
                                    logits_to_keep = max(1, logits_to_keep)
                                    teacher_inputs_for_logprobs['logits_to_keep'] = logits_to_keep
                        with self._opd_teacher_forward_timer():
                            teacher_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                                self.teacher_model, teacher_inputs_for_logprobs, compute_entropy=False)
                        encoded_inputs['teacher_per_token_logps'] = teacher_per_token_logps
                        logger.info(f"[DEBUG training_step] teacher_per_token_logps shape: {teacher_per_token_logps.shape}")

            else:
                # Off-policy: use dataset responses, encode full messages
                data_source = DataSource.DATASET
                total_length = self.template.max_length + self.max_completion_length
                with self._template_context(self.template, max_length=total_length):
                    encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)

            # Mark data source for downstream processing (e.g., conditional SFT loss)
            encoded_inputs['_data_source'] = data_source

        with self.template.forward_context(self.model, encoded_inputs):
            loss = HFSFTTrainer.training_step(self, model, encoded_inputs, num_items_in_batch)
        return loss

    def prediction_step(self, model, inputs, *args, **kwargs):
        # Prediction uses full messages
        encoded_inputs = self._prepare_batch_inputs(inputs, encode_prompt_only=False)
        with self.template.forward_context(self.model, encoded_inputs):
            return super().prediction_step(model, encoded_inputs, *args, **kwargs)

    @contextmanager
    def offload_context(self):
        """Context manager for offloading model and optimizer during vLLM inference

        This offloads:
        - Student model (self.model)
        - Optimizer states

        to CPU to free up GPU memory for vLLM engine.
        """
        if self.args.offload_model:
            self.offload_model(self.accelerator.unwrap_model(self.model))
        if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
            self.offload_optimizer()

        try:
            yield
        finally:
            # reload (load back) model when exiting context
            if self.args.offload_model:
                self.load_model(self.accelerator.unwrap_model(self.model))
            if getattr(self, 'optimizer', None) and self.args.offload_optimizer:
                self.load_optimizer()

    def _get_random_num(self) -> float:
        """
        Generate a deterministic random number.

        Uses an isolated Random instance to avoid interfering with the global
        random state, ensuring thread-safety and consistent behavior across processes.

        Returns:
            float: A random number in the range [0.0, 1.0).
        """
        seed = int(getattr(self.args, 'seed', 0))
        seed += int(self.state.global_step)
        rng = random.Random(seed)
        return rng.random()

    @contextmanager
    def load_teacher_model_context(self):
        """
        Context manager to load and offload the teacher model with memory and timing profiling.
        """
        if not self.args.offload_teacher_model:
            yield
            return

        self.load_model(self.accelerator.unwrap_model(self.teacher_model))
        yield
        self.offload_model(self.accelerator.unwrap_model(self.teacher_model))

    def _prepare_liger_loss(self):
        """Initialize liger loss if enabled."""
        args = self.args
        self.use_liger_gkd_loss = False
        if getattr(args, 'use_liger_kernel', False):
            if not _liger_kernel_available:
                raise ImportError(
                    'Liger kernel is not installed. Please install liger-kernel by running: pip install liger-kernel')
            assert self.args.sft_alpha == 0, 'SFT loss is not supported with liger loss'

            self.liger_jsd_loss = LigerFusedLinearJSDLoss(
                beta=self.beta,
                ignore_index=-100,
                temperature=self.temperature,
                compiled=False,
            )
            self.use_liger_gkd_loss = True

    def _prepare_logging(self):
        """Initialize logging components for on-policy rollout tracking."""
        args = self.args
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = getattr(args, 'wandb_log_unique_prompts', False)
        self.jsonl_writer = JsonlWriter(os.path.join(self.args.output_dir, 'completions.jsonl'))

        # Initialize logs deque for storing rollout data (aligned with GRPO)
        self._logs = {
            'prompt': deque(),
            'completion': deque(),
        }

    def _apply_chat_template_to_messages_list(self, messages_list: DataType):
        """Convert messages list to prompt text list using template (aligned with GRPO)."""
        prompts_text = []
        for messages in messages_list:
            remove_response(messages)
            template_inputs = TemplateInputs.from_dict({'messages': messages})
            res = self.template.encode(template_inputs)
            prompts_text.append(self.template.safe_decode(res['input_ids']))
        return prompts_text

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log method to include completion table logging (aligned with GRPO)."""
        # Compute metrics from _metrics dictionary (exactly like GRPO trainer line 1844)
        mode = 'train' if self.model.training else 'eval'
        # Average the metrics - exactly like GRPO trainer (line 1844)
        # Note: GRPO doesn't check if lists are empty, it just computes averages
        # Empty lists will cause ZeroDivisionError, but that's handled by only including non-empty lists
        metrics = {
            key: sum(val) / len(val) 
            for key, val in self._metrics[mode].items() 
            if len(val) > 0
        }
        
        # Add prefix for eval mode to match parent's format (exactly like GRPO trainer line 1848-1849)
        if mode == 'eval':
            metrics = {f'eval_{key}': val for key, val in metrics.items()}
        
        # Add metrics to logs before calling parent (exactly like GRPO trainer line 1851)
        logs.update(metrics)
        
        # Clear metrics after logging (exactly like GRPO trainer line 1856)
        self._metrics[mode].clear()
        
        # Call parent log method
        import transformers
        from packaging import version
        if version.parse(transformers.__version__) >= version.parse('4.47.0.dev0'):
            super().log(logs, start_time)
        else:
            super().log(logs)

        # Log completions table if we have data (only for on-policy generations)
        if self.accelerator.is_main_process and self.log_completions and len(self._logs['prompt']) > 0:
            seen_nums = len(self._logs['prompt'])
            table = {
                'step': [str(self.state.global_step)] * seen_nums,
                'prompt': list(self._logs['prompt'])[:seen_nums],
                'completion': list(self._logs['completion'])[:seen_nums],
            }

            # Write to jsonl
            self.jsonl_writer.append(table)

            self._logs['prompt'].clear()
            self._logs['completion'].clear()
            # Log to wandb if enabled
            report_to_wandb = self.args.report_to and 'wandb' in self.args.report_to and wandb.run is not None
            if report_to_wandb:
                wandb_table = table.copy()
                import pandas as pd
                df = pd.DataFrame(wandb_table)
                if self.wandb_log_unique_prompts:
                    df = df.drop_duplicates(subset=['prompt'])
                wandb.log({'completions': wandb.Table(dataframe=df)})

            # Log to swanlab if enabled
            report_to_swanlab = self.args.report_to and 'swanlab' in self.args.report_to and swanlab.get_run(
            ) is not None
            if report_to_swanlab:
                headers = list(table.keys())
                rows = []
                for i in range(len(table['step'])):
                    row = [table[header][i] for header in headers]
                    rows.append(row)
                swanlab.log({'completions': swanlab.echarts.Table().add(headers, rows)})

    # ========== Methods reused from GRPO trainer ==========
    # These methods are copied from GRPOTrainer to ensure consistency
    # Source: swift/trainers/rlhf_trainer/grpo_trainer.py
    
    def _prepare_inputs(self, inputs):
        """Override _prepare_inputs to preserve old_per_token_logps and _data_source."""
        # Call parent _prepare_inputs but preserve on-policy distillation fields
        old_per_token_logps = inputs.get('old_per_token_logps')
        data_source = inputs.get('_data_source')
        
        inputs = super()._prepare_inputs(inputs)
        
        # Restore on-policy distillation fields if they were present
        if old_per_token_logps is not None:
            inputs['old_per_token_logps'] = old_per_token_logps
        if data_source is not None:
            inputs['_data_source'] = data_source
        
        return inputs
    
    def _prepare_model_inputs(self, inputs: DataType):
        """Prepare model inputs, reused from GRPO trainer (grpo_trainer.py line 2543)."""
        return {
            k: v
            for k, v in inputs.items() if k not in [
                'logits_to_keep', 'completion_mask', 'ref_per_token_logps', 'advantages', 'old_per_token_logps', 'labels',
                'truncated_mask', 'seq_lengths', 'num_items_in_batch', 'rollout_per_token_logps'
            ]
        }
    
    @patch_profiling_decorator
    def _get_per_token_logps_and_entropies(self,
                                           model,
                                           inputs,
                                           compute_entropy=False):
        """
        Compute per-token log probabilities and entropies.
        
        Reused from GRPO trainer (grpo_trainer.py lines 1607-1624).
        This ensures consistency with GRPO's handling of padding_free, SP, etc.
        """
        batch_size = inputs['seq_lengths'].shape[0] if self.template.padding_free else inputs['input_ids'].shape[0]
        mode = 'train' if self.model.training else 'eval'
        expected_bs = self.args.per_device_train_batch_size if mode == 'train' else self.args.per_device_eval_batch_size
        should_chunk = getattr(self, 'dynamic_num_samples', False) and any(gather_object([batch_size > expected_bs]))
        if not should_chunk:
            return self._get_per_token_logps_and_entropies_single(model, inputs, compute_entropy=compute_entropy)
        else:
            return self._get_per_token_logps_and_entropies_chunked(model, inputs, compute_entropy=compute_entropy)
    
    def _get_per_token_logps_and_entropies_single(self,
                                                  model,
                                                  inputs,
                                                  compute_entropy=False):
        """
        Single batch version, reused from GRPO trainer (grpo_trainer.py lines 1626-1676).
        """
        logits_to_keep = inputs['logits_to_keep']
        input_ids = inputs['input_ids']
        is_padding_free = self.template.padding_free
        use_sp = self.template.sequence_parallel_size > 1
        
        if is_padding_free:
            original_seq_lengths = inputs.get('seq_lengths')
            batch_size = original_seq_lengths.shape[0]
        
        unwrapped_model = self.accelerator.unwrap_model(model)
        if is_peft_model(unwrapped_model):
            parameters = inspect.signature(unwrapped_model.base_model.model.forward).parameters
        else:
            parameters = inspect.signature(unwrapped_model.forward).parameters
        use_local_entropy = not hasattr(super(), '_get_per_token_logps_and_entropies') and compute_entropy
        
        can_use_super = (not getattr(self, 'is_multimodal', False) and 'logits_to_keep' in parameters 
                        and not use_local_entropy and not is_padding_free and not use_sp)
        
        if can_use_super:
            if hasattr(super(), '_get_per_token_logps_and_entropies'):
                logps, entropies = super()._get_per_token_logps_and_entropies(
                    model, input_ids, inputs['attention_mask'], logits_to_keep, compute_entropy=compute_entropy)
            else:
                logps = super()._get_per_token_logps(model, input_ids, inputs['attention_mask'], logits_to_keep)
                entropies = None
        elif use_sp:
            logps, entropies = self._get_logps_via_sp(
                model, inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy)
        else:
            logps, entropies = self._get_logps_via_local_forward(
                model, inputs, logits_to_keep, input_ids, compute_entropy=compute_entropy)
        
        if is_padding_free:
            logps, entropies = self._unpad_logps_and_entropies(logps, entropies, logits_to_keep, batch_size,
                                                               original_seq_lengths, compute_entropy)
        
        return logps, entropies
    
    def _get_logps_via_local_forward(self,
                                     model: torch.nn.Module,
                                     inputs: DataType,
                                     logits_to_keep: int,
                                     input_ids: torch.Tensor,
                                     compute_entropy: bool = False):
        """
        Get per token logps via local forward pass, reused from GRPO trainer (grpo_trainer.py lines 1563-1604).
        """
        from trl.trainer.utils import selective_log_softmax, entropy_from_logits
        
        # Ensure logits_to_keep is a scalar integer (not a tensor)
        # In padding_free mode, logits_to_keep might be a tensor representing total completion tokens
        # In non-padding_free mode, it should be a scalar
        if isinstance(logits_to_keep, torch.Tensor):
            if logits_to_keep.numel() == 1:
                logits_to_keep = logits_to_keep.item()
            else:
                # If it's a multi-element tensor, it likely represents total completion tokens in padding_free mode
                # Use the sum or max depending on the context
                # For padding_free, logits_to_keep should be the total number of completion tokens
                logits_to_keep = logits_to_keep.sum().item() if self.template.padding_free else logits_to_keep.max().item()
        logits_to_keep = int(logits_to_keep)
        
        model_inputs = self._prepare_model_inputs(inputs)
        if 'logits_to_keep' in self.model_kwarg_keys:
            model_inputs['logits_to_keep'] = logits_to_keep + 1
        
        logits = model(**model_inputs).logits
        
        logits = logits[:, -(logits_to_keep + 1):-1, :] / self.temperature
        input_ids_for_logps = input_ids[:, -logits_to_keep:]
        
        is_padding_free = self.template.padding_free
        if is_padding_free:
            logits_rmpad = logits.squeeze(0)
            input_ids_rmpad = input_ids_for_logps.squeeze(0)
            logps = selective_log_softmax(logits_rmpad, input_ids_rmpad)
            logps = logps.unsqueeze(0)
            if compute_entropy:
                entropies = entropy_from_logits(logits_rmpad).unsqueeze(0)
            else:
                entropies = None
        else:
            logps = selective_log_softmax(logits, input_ids_for_logps)
            if compute_entropy:
                entropies = entropy_from_logits(logits)
            else:
                entropies = None
        
        return logps, entropies
    
    def _get_logps_via_sp(self,
                          model: torch.nn.Module,
                          inputs: DataType,
                          logits_to_keep: int,
                          input_ids: torch.Tensor,
                          compute_entropy: bool = False):
        """
        Get per token logps via sequence parallel, reused from GRPO trainer (grpo_trainer.py lines 1464-1561).
        """
        from trl.trainer.utils import selective_log_softmax, entropy_from_logits
        from swift.trainers.sequence_parallel.utils import GatherLoss
        from swift.trainers.sequence_parallel import sequence_parallel
        
        model_inputs = self._prepare_model_inputs(inputs)
        sequence_parallel.prepare_inputs(model_inputs)
        with self._template_context(self.template, inputs):
            output = model(**model_inputs)
            logits = output.logits
        # split input_ids to labels
        position_ids = sequence_parallel.real_position_ids
        _, _, labels, _, _, _, _ = sequence_parallel.pad_and_split_inputs(
            None, None, input_ids.clone(), None, None, None, real_position_ids=position_ids)
        
        labels = torch.where(labels == -100, self.processing_class.pad_token_id, labels)
        logits = logits / self.temperature
        per_token_logps = selective_log_softmax(logits, labels)
        entropies = None
        per_token_logps, _ = GatherLoss.apply(per_token_logps, labels, 1, position_ids)
        if compute_entropy:
            entropies = entropy_from_logits(logits)
            entropies, _ = GatherLoss.apply(entropies, labels, 1, position_ids)
        
        if self.template.padding_free:
            # In padding_free mode, we need to extract completion tokens from gathered data.
            # The behavior differs based on rp_world_size:
            # - rp_world_size > 1: Each sequence is padded to world_size * 2 multiple (per-sequence padding)
            # - rp_world_size == 1: Entire data is padded to world_size multiple (end padding only)
            seq_lengths = inputs['seq_lengths']
            batch_size = seq_lengths.shape[0]
            rp_world_size = sequence_parallel.rp_world_size
            
            from swift.utils import get_cu_seqlens_from_position_ids
            
            if rp_world_size > 1:
                # With ring parallel: GatherLoss pads each sequence to world_size * 2 multiple
                # Data layout after gather: [seq1_data, seq1_padding, seq2_data, seq2_padding, ...]
                # - Original data is at [offset:offset+orig_len]
                # - Padding is at [offset+orig_len:offset+padded_len]
                
                # Get original sequence boundaries (before padding)
                cu_seqlens_orig = get_cu_seqlens_from_position_ids(position_ids)
                
                # Get padded sequence boundaries (for offset calculation)
                padded_position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
                cu_seqlens_padded = get_cu_seqlens_from_position_ids(padded_position_ids)
                
                result_logps = []
                result_entropies = [] if compute_entropy else None
                gathered_logps = per_token_logps.squeeze(0)
                gathered_entropies = entropies.squeeze(0) if compute_entropy else None
                
                offset = 0
                for i in range(batch_size):
                    # Original sequence length (before SP padding)
                    orig_len = (cu_seqlens_orig[i + 1] - cu_seqlens_orig[i]).item()
                    # Padded sequence length (multiple of world_size * 2)
                    padded_len = (cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]).item()
                    # Actual completion tokens for this sequence
                    actual_len = seq_lengths[i].item()
                    
                    # Extract the last `actual_len` tokens from this sequence's ORIGINAL data region
                    # Due to label shifting (roll -1), per_token_logps[i] predicts token i+1
                    # So completion tokens [prompt_len, total_len) have logps at [prompt_len-1, total_len-1)
                    seq_start = offset + orig_len - actual_len - 1
                    seq_end = offset + orig_len - 1
                    result_logps.append(gathered_logps[seq_start:seq_end])
                    if compute_entropy:
                        result_entropies.append(gathered_entropies[seq_start:seq_end])
                    
                    # Use padded_len for offset because gathered data includes padding
                    offset += padded_len
                
                per_token_logps = torch.cat(result_logps).unsqueeze(0)
                if compute_entropy:
                    entropies = torch.cat(result_entropies).unsqueeze(0)
            else:
                # Without ring parallel (rp_world_size == 1): Simple gather with end padding only
                # Use input_ids length directly as the authoritative original length
                original_total_len = input_ids.shape[-1]
                # Due to label shifting (roll -1), per_token_logps[i] predicts token i+1.
                start_idx = original_total_len - logits_to_keep - 1
                end_idx = original_total_len - 1
                per_token_logps = per_token_logps[:, start_idx:end_idx]
                if compute_entropy:
                    entropies = entropies[:, start_idx:end_idx]
        else:
            per_token_logps = per_token_logps[:, -logits_to_keep - 1:-1]
            if compute_entropy:
                entropies = entropies[:, -logits_to_keep - 1:-1]
        
        return per_token_logps, entropies
    
    def _get_per_token_logps_and_entropies_chunked(self,
                                                   model,
                                                   inputs,
                                                   compute_entropy=False):
        """
        Chunked version for large batches, reused from GRPO trainer pattern.
        For simplicity, delegate to single version for now.
        """
        return self._get_per_token_logps_and_entropies_single(model, inputs, compute_entropy)
    
    def _unpad_logps_and_entropies(self,
                                   logps: torch.Tensor,
                                   entropies: Optional[torch.Tensor],
                                   logits_to_keep: int,
                                   batch_size: int,
                                   seq_lengths: torch.Tensor,
                                   compute_entropy: bool = False):
        """
        Restore logps from rmpad format, reused from GRPO trainer (grpo_trainer.py lines 1433-1462).
        """
        from swift.trainers.rlhf_trainer.utils import pad_logps_back_to_batch
        
        logps, _ = pad_logps_back_to_batch(
            logps_rmpad=logps, logits_to_keep=logits_to_keep, batch_size=batch_size, seq_lengths=seq_lengths)
        
        if compute_entropy and entropies is not None:
            entropies, _ = pad_logps_back_to_batch(
                logps_rmpad=entropies, logits_to_keep=logits_to_keep, batch_size=batch_size, seq_lengths=seq_lengths)
        
        return logps, entropies

    def _get_sampling_logprobs(self, model, inputs):
        """Get per-token logprobs from model at sampling time.
        
        Reuses GRPO's _get_per_token_logps_and_entropies method.
        This should match GRPO's _prepare_batch_inputs logic (grpo_trainer.py lines 836-848).
        """
        # Prepare logits_to_keep if needed (same as in _compute_on_policy_distillation_loss)
        # Note: _get_per_token_logps_and_entropies always needs logits_to_keep, even if use_logits_to_keep=False
        inputs_for_logprobs = inputs.copy()
        if 'logits_to_keep' not in inputs_for_logprobs:
            # Use GRPO's exact calculation method (grpo_trainer.py line 836)
            # This ensures consistency with GRPO's behavior
            labels = inputs_for_logprobs.get('labels')
            if labels is not None:
                # GRPO's calculation: (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
                # This finds the first non-padding token position and computes completion length
                non_padding_mask = torch.ne(labels, -100).int()
                if non_padding_mask.any():
                    # Find first non-padding position for each sequence
                    first_non_padding = non_padding_mask.argmax(-1)
                    # Compute logits_to_keep: total_length - first_non_padding_position
                    # This gives the length of the completion part
                    logits_to_keep = (labels.shape[-1] - first_non_padding).max().item()
                    # Ensure logits_to_keep is at least 1
                    logits_to_keep = max(1, logits_to_keep)
                    inputs_for_logprobs['logits_to_keep'] = logits_to_keep
                    logger.info(f"[DEBUG _get_sampling_logprobs] Computed logits_to_keep={logits_to_keep} from labels shape={labels.shape}, first_non_padding={first_non_padding.tolist()}")
                else:
                    # Fallback to prepare_logits_to_keep if no completion tokens found
                    self.prepare_logits_to_keep(inputs_for_logprobs)
            else:
                # Fallback to prepare_logits_to_keep if no labels
                self.prepare_logits_to_keep(inputs_for_logprobs)
        
        # Use GRPO's method for consistency
        per_token_logps, _ = self._get_per_token_logps_and_entropies(
            model, inputs_for_logprobs, compute_entropy=False)
        # logger.info(f"[DEBUG _get_sampling_logprobs] per_token_logps shape={per_token_logps.shape}, logits_to_keep={inputs_for_logprobs.get('logits_to_keep')}")
        return per_token_logps
    
    def _get_vllm_sampling_logprobs(self, model, inputs):
        """Get per-token logprobs from vLLM rollout outputs."""
        # For vLLM, logprobs should be extracted from rollout outputs
        # This is a simplified version - in practice, vLLM returns logprobs in the response
        # For now, we compute them from the model
        return self._get_sampling_logprobs(model, inputs)

    def _compute_on_policy_distillation_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss using on-policy distillation algorithm.
        
        Reuses GRPO's methods for:
        1. Computing per-token logprobs (_get_per_token_logps_and_entropies)
        2. Importance sampling computation (same logic as GRPO)
        3. Loss normalization (same as GRPO)
        
        Algorithm (from on-policy.md):
        - reverse_kl = sampled_logprobs - teacher_logprobs
        - advantages = -reverse_kl
        - loss = importance_sampling(advantages)
        """
        # Get old logprobs (from sampling time) - this tells us the correct logits_to_keep
        old_per_token_logps = inputs['old_per_token_logps']
        
        # Use old_per_token_logps shape to determine logits_to_keep
        # This ensures consistency with sampling time calculation
        if old_per_token_logps.dim() == 2:
            logits_to_keep_from_old = old_per_token_logps.shape[1]
        else:
            # Fallback: compute from labels using GRPO's method
            labels = inputs['labels']
            logits_to_keep_from_old = (labels.shape[-1] - (torch.ne(labels, -100).int().argmax(-1))).max().item()
        
        # Ensure logits_to_keep is set correctly in inputs for _get_per_token_logps_and_entropies
        # Don't use prepare_logits_to_keep as it modifies labels, which we need intact
        inputs_for_logprobs = inputs.copy()
        inputs_for_logprobs['logits_to_keep'] = logits_to_keep_from_old
        
        logger.info(f"[DEBUG _compute_on_policy_distillation_loss] Using logits_to_keep={logits_to_keep_from_old} from old_per_token_logps shape={old_per_token_logps.shape}")
        # Print prompt and generated response for debugging - show full content
        if self.accelerator.is_main_process and self.state.global_step % 10 == 0:  # Print every 10 steps
            input_ids = inputs.get('input_ids')
            labels = inputs.get('labels')
            if input_ids is not None and labels is not None:
                # Find the boundary between prompt and completion
                # labels == -100 indicates prompt tokens, labels != -100 indicates completion tokens
                completion_mask = labels != -100
                if completion_mask.any():
                    # Get the first completion token position
                    first_completion_idx = completion_mask.int().argmax(-1)
                    if input_ids.dim() == 2:
                        # Batch format: [batch_size, seq_len]
                        batch_size = input_ids.shape[0]
                        for i in range(batch_size):  # Print all samples in batch
                            seq_len = input_ids.shape[1]
                            first_comp = first_completion_idx[i].item() if first_completion_idx.dim() > 0 else first_completion_idx.item()
                            
                            # Extract prompt and completion token ids
                            prompt_ids = input_ids[i, :first_comp].cpu().tolist()
                            completion_ids = input_ids[i, first_comp:].cpu().tolist()
                            
                            # Decode to text - show full content without truncation
                            try:
                                prompt_text = self.template.safe_decode(prompt_ids) if prompt_ids else ""
                                completion_text = self.template.safe_decode(completion_ids) if completion_ids else ""
                                
                                logger.info(f"[DEBUG OPD Training] Step {self.state.global_step}, Sample {i}:")
                                logger.info(f"  Prompt ({len(prompt_ids)} tokens):\n{prompt_text}")
                                logger.info(f"  Generated Response ({len(completion_ids)} tokens):\n{completion_text}")
                                logger.info("  " + "="*80)  # Separator line
                            except Exception as e:
                                logger.warning(f"[DEBUG OPD Training] Failed to decode tokens: {e}")
                    elif input_ids.dim() == 1:
                        # Single sequence format
                        first_comp = first_completion_idx.item() if isinstance(first_completion_idx, torch.Tensor) else first_completion_idx
                        prompt_ids = input_ids[:first_comp].cpu().tolist()
                        completion_ids = input_ids[first_comp:].cpu().tolist()
                        
                        try:
                            prompt_text = self.template.safe_decode(prompt_ids) if prompt_ids else ""
                            completion_text = self.template.safe_decode(completion_ids) if completion_ids else ""
                            
                            logger.info(f"[DEBUG OPD Training] Step {self.state.global_step}:")
                            logger.info(f"  Prompt ({len(prompt_ids)} tokens):\n{prompt_text}")
                            logger.info(f"  Generated Response ({len(completion_ids)} tokens):\n{completion_text}")
                            logger.info("  " + "="*80)  # Separator line
                        except Exception as e:
                            logger.warning(f"[DEBUG OPD Training] Failed to decode tokens: {e}")
        
        # Prepare inputs - reuse GRPO's pattern
        model_inputs = {k: v for k, v in inputs.items() if k not in {'prompt', 'labels'}}
        use_logits_to_keep = self.get_use_logits_to_keep(True)
        if use_logits_to_keep:
            model_inputs['logits_to_keep'] = logits_to_keep_from_old
        
        # Compute current policy logprobs - reuse GRPO's method
        # Use inputs_for_logprobs which has the correct logits_to_keep
        per_token_logps, _ = self._get_per_token_logps_and_entropies(model, inputs_for_logprobs, compute_entropy=False)
        
        # Get teacher logprobs - should already be computed in compute_loss to avoid redundant computation
        # Check if it was computed during teacher sampling (seq_kd=True) or in compute_loss (use_on_policy_distillation=True)
        if 'teacher_per_token_logps' in inputs:
            # Reuse teacher logprobs (either from sampling or from compute_loss)
            teacher_per_token_logps = inputs['teacher_per_token_logps']
            # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] Reusing teacher_per_token_logps, shape={teacher_per_token_logps.shape}")
        else:
            # Fallback: compute teacher logprobs here if not already computed
            # This should not happen if compute_loss is called correctly
            logger.warning("[DEBUG _compute_on_policy_distillation_loss] teacher_per_token_logps not found, computing now (this should not happen)")
            model_inputs.pop('labels', None)
            load_context = self.load_teacher_model_context() if self.args.offload_teacher_model else nullcontext()
            with torch.no_grad(), load_context:
                teacher_inputs = inputs_for_logprobs.copy()
                teacher_inputs.pop('old_per_token_logps', None)
                teacher_inputs.pop('_data_source', None)
                # Ensure teacher_inputs has the correct logits_to_keep
                teacher_inputs['logits_to_keep'] = logits_to_keep_from_old
                with self._opd_teacher_forward_timer():
                    teacher_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.teacher_model, teacher_inputs, compute_entropy=False)
                logger.info(f"[DEBUG _compute_on_policy_distillation_loss] Computed teacher_per_token_logps, shape={teacher_per_token_logps.shape}")
        
        # Ensure per_token_logps and teacher_per_token_logps shapes match
        if per_token_logps.shape != teacher_per_token_logps.shape:
            # Align shapes by padding or truncating
            if per_token_logps.shape[1] < teacher_per_token_logps.shape[1]:
                # Pad per_token_logps
                padding_size = teacher_per_token_logps.shape[1] - per_token_logps.shape[1]
                padding = torch.zeros(
                    per_token_logps.shape[0],
                    padding_size,
                    device=per_token_logps.device,
                    dtype=per_token_logps.dtype
                )
                per_token_logps = torch.cat([per_token_logps, padding], dim=1)
            elif per_token_logps.shape[1] > teacher_per_token_logps.shape[1]:
                # Truncate per_token_logps
                per_token_logps = per_token_logps[:, :teacher_per_token_logps.shape[1]]
        
        # Recreate completion_mask AFTER shape alignment to match final aligned shape
        # Use the same logits_to_keep that was used for computing logprobs
        # Note: No need to shift labels here - completion_mask just marks which positions need loss computation
        # The alignment between logits and labels is already handled in _get_per_token_logps_and_entropies
        labels = inputs['labels']
        final_seq_len = per_token_logps.shape[1]
        
        # Create completion_mask with the same shape as aligned per_token_logps
        # Use GRPO's exact pattern: labels[:, -logits_to_keep:] != -100
        # This matches GRPO trainer's implementation (grpo_trainer.py line 842)
        if labels.shape[1] >= final_seq_len:
            completion_mask = labels[:, -final_seq_len:] != -100
        else:
            # If labels is shorter, pad with False (no completion tokens)
            padding_size = final_seq_len - labels.shape[1]
            completion_mask_base = labels != -100
            padding = torch.zeros(
                completion_mask_base.shape[0],
                padding_size,
                device=completion_mask_base.device,
                dtype=completion_mask_base.dtype
            ).bool()
            completion_mask = torch.cat([completion_mask_base, padding], dim=1)
        
        # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] completion_mask created: shape={completion_mask.shape}, sum={completion_mask.sum().item()}, final_seq_len={final_seq_len}")
        
        # Ensure shapes match for old_per_token_logps BEFORE computing reverse_kl
        if old_per_token_logps.shape != teacher_per_token_logps.shape:
            if self.template.padding_free:
                if old_per_token_logps.dim() == 2 and old_per_token_logps.shape[0] == 1:
                    old_per_token_logps = old_per_token_logps.view(teacher_per_token_logps.shape)
            else:
                if old_per_token_logps.shape[1] < teacher_per_token_logps.shape[1]:
                    padding = torch.zeros(
                        old_per_token_logps.shape[0],
                        teacher_per_token_logps.shape[1] - old_per_token_logps.shape[1],
                        device=old_per_token_logps.device
                    )
                    old_per_token_logps = torch.cat([old_per_token_logps, padding], dim=1)
                elif old_per_token_logps.shape[1] > teacher_per_token_logps.shape[1]:
                    old_per_token_logps = old_per_token_logps[:, :teacher_per_token_logps.shape[1]]
        
        # Debug: Print shapes and sample values before computing reverse_kl
        logger.info(f"[DEBUG _compute_on_policy_distillation_loss] old_per_token_logps shape={old_per_token_logps.shape}, teacher_per_token_logps shape={teacher_per_token_logps.shape}")
        # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] completion_mask shape={completion_mask.shape}, completion_mask.sum()={completion_mask.sum().item()}")
        # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] old_per_token_logps sample (first 5): {old_per_token_logps[0, :5].tolist()}")
        # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] teacher_per_token_logps sample (first 5): {teacher_per_token_logps[0, :5].tolist()}")
        
        # Compute reverse KL = sampled_logprobs - teacher_logprobs (as per on-policy.md)
        # According to on-policy.md pseudocode: reverse_kl = sampled_logprobs - teacher_logprobs
        # where sampled_logprobs = old_per_token_logps (from sampling time)
        reverse_kl = old_per_token_logps - teacher_per_token_logps
        
        # Debug: Print reverse_kl values
        # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] reverse_kl shape={reverse_kl.shape}, reverse_kl sample (first 5): {reverse_kl[0, :5].tolist()}")
        # logger.info(f"[DEBUG _compute_on_policy_distillation_loss] reverse_kl[completion_mask] shape={reverse_kl[completion_mask].shape}, mean={reverse_kl[completion_mask].mean().item()}")
        
        # Compute advantage = -reverse_kl (as per on-policy.md)
        # According to on-policy.md pseudocode: advantages = -reverse_kl
        # Note: This means:
        # - When reverse_kl > 0 (sampled student better than teacher), advantages < 0
        # - When reverse_kl < 0 (sampled student worse than teacher), advantages > 0
        # This is correct for policy gradient: we want to maximize advantages
        advantages = -reverse_kl
        
        # Ensure shapes match for old_per_token_logps with per_token_logps (for importance sampling)
        if old_per_token_logps.shape != per_token_logps.shape:
            if self.template.padding_free:
                if old_per_token_logps.dim() == 2 and old_per_token_logps.shape[0] == 1:
                    old_per_token_logps = old_per_token_logps.view(per_token_logps.shape)
            else:
                if old_per_token_logps.shape[1] < per_token_logps.shape[1]:
                    padding = torch.zeros(
                        old_per_token_logps.shape[0],
                        per_token_logps.shape[1] - old_per_token_logps.shape[1],
                        device=old_per_token_logps.device
                    )
                    old_per_token_logps = torch.cat([old_per_token_logps, padding], dim=1)
                elif old_per_token_logps.shape[1] > per_token_logps.shape[1]:
                    old_per_token_logps = old_per_token_logps[:, :per_token_logps.shape[1]]
        
        # Compute importance sampling weights - reuse GRPO's logic exactly
        # (from grpo_trainer.py lines 1127-1144)
        log_ratio = per_token_logps - old_per_token_logps
        importance_sampling_level = getattr(self.args, 'importance_sampling_level', 'token')
        
        if importance_sampling_level == 'token':
            log_importance_weights = log_ratio
        elif importance_sampling_level in ['sequence', 'sequence_token']:
            seq_level_log_weights = ((log_ratio * completion_mask.float()).sum(-1)
                                     / completion_mask.float().sum(-1).clamp(min=1.0)).unsqueeze(-1)
            if importance_sampling_level == 'sequence':
                log_importance_weights = seq_level_log_weights
            else:
                # GSPO-token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
                seq_level_log_weight = seq_level_log_weights.detach()
                log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        else:
            log_importance_weights = log_ratio  # Default to token-level
        
        importance_weights = torch.exp(log_importance_weights)
        
        # Compute loss according to on-policy distillation formula:
        # loss = -(prob_ratio * advantages).sum()
        # where prob_ratio = exp(target_logprobs - sampling_logprobs) = importance_weights
        # This is the standard importance-weighted policy gradient loss
        per_token_loss = -importance_weights * advantages
        
        # Apply completion mask and sum directly (no averaging)
        # According to the formula: loss = -(prob_ratio * advantages).sum()
        loss = (per_token_loss * completion_mask.float()).sum()
        
        # Log metrics (aligned with GRPO trainer)
        mode = 'train' if self.model.training else 'eval'
        with torch.no_grad():
            # Ensure completion_mask shape matches reverse_kl shape
            if completion_mask.shape != reverse_kl.shape:
                # Align completion_mask to match reverse_kl shape
                if completion_mask.shape[1] < reverse_kl.shape[1]:
                    padding_size = reverse_kl.shape[1] - completion_mask.shape[1]
                    padding = torch.zeros(
                        completion_mask.shape[0],
                        padding_size,
                        device=completion_mask.device,
                        dtype=completion_mask.dtype
                    )
                    completion_mask = torch.cat([completion_mask, padding], dim=1)
                elif completion_mask.shape[1] > reverse_kl.shape[1]:
                    completion_mask = completion_mask[:, :reverse_kl.shape[1]]
            
            # Compute reward as mean reverse_kl (reverse_kl is the reward in on-policy distillation)
            reward_mean = reverse_kl[completion_mask].mean().item()
            advantage_mean = advantages[completion_mask].mean().item()
            importance_weight_mean = importance_weights[completion_mask].mean().item()
            
            # Print reward and advantage for debugging
            logger.info(f"[On-Policy Distillation] reward={reward_mean:.6f}, advantage={advantage_mean:.6f}, importance_weight={importance_weight_mean:.6f}")
            
            self._metrics[mode]['reward'].append(reward_mean)
            self._metrics[mode]['reverse_kl'].append(reward_mean)
            self._metrics[mode]['advantage'].append(advantage_mean)
            self._metrics[mode]['importance_weight'].append(importance_weight_mean)
        
        if return_outputs:
            # Re-compute outputs if needed
            outputs_student = model(**model_inputs)
            return (loss, outputs_student)
        else:
            return loss
    
    def _compute_per_token_logps_from_outputs(self, outputs, labels, is_teacher=False):
        """Helper method to compute per-token logprobs from model outputs."""
        from trl.trainer.utils import selective_log_softmax
        
        if not self.is_encoder_decoder:
            shifted_labels = torch.roll(labels, shifts=-1, dims=1)
        else:
            shifted_labels = labels
        
        logits = outputs.logits
        if logits.shape[1] != shifted_labels.shape[1]:
            logits = logits[:, -shifted_labels.shape[1]:]
        
        # Fix vocab_size mismatch for teacher
        if is_teacher and hasattr(self, 'model'):
            student_outputs = self.model(**{k: v for k, v in outputs.__dict__.items() if k in ['input_ids', 'attention_mask']})
            student_logits = student_outputs.logits if hasattr(student_outputs, 'logits') else None
            if student_logits is not None and student_logits.shape[-1] != logits.shape[-1]:
                stu_dim = student_logits.shape[-1]
                tea_dim = logits.shape[-1]
                if stu_dim < tea_dim:
                    logits = F.pad(logits, (0, tea_dim - stu_dim), 'constant', 0)
                elif stu_dim > tea_dim:
                    logits = F.pad(logits, (0, stu_dim - tea_dim), 'constant', 0)
        
        mask = shifted_labels != -100
        if self.template.padding_free:
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = shifted_labels.view(-1)
            mask_flat = labels_flat != -100
            per_token_logps_flat = selective_log_softmax(logits_flat, labels_flat)
            per_token_logps_flat[~mask_flat] = 0
            per_token_logps = per_token_logps_flat.view(shifted_labels.shape)
        else:
            per_token_logps = selective_log_softmax(logits, shifted_labels)
            per_token_logps[~mask] = 0
        
        return per_token_logps
