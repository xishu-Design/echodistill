from __future__ import annotations

import json
import time
import importlib
import gc
import inspect
import random
import re
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from sdpoa.training.losses import distill_kl_loss, grpo_policy_loss
from sdpoa.utils.audio import load_audio_mono


class SampleSkipError(RuntimeError):
    pass


@dataclass
class StepOutput:
    total: torch.Tensor
    policy: torch.Tensor
    distill: torch.Tensor
    feedback: torch.Tensor
    reward_mean: float
    skipped_samples: int
    valid_samples: int


class DistillationTrainer:
    def __init__(self, cfg, bundle, device: torch.device):
        self.cfg = cfg
        self.student_device = bundle.student_device or device
        self.teacher_device = bundle.teacher_device or self.student_device
        self.device = self.student_device
        self.student = bundle.student_model
        self.teacher = bundle.teacher_model
        self.tokenizer = bundle.tokenizer
        self.processor = bundle.processor
        self._strict_audio_distillation = self.processor is not None and not self.cfg.model.force_text_only_omni

        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=cfg.train.learning_rate,
            weight_decay=cfg.train.weight_decay,
        )

        # Multimodal audio inputs can change effective sequence shapes between forward
        # and checkpoint recomputation, which breaks backward with CheckpointError.
        if cfg.train.gradient_checkpointing:
            if self._strict_audio_distillation:
                print(
                    "[warn] gradient checkpointing requested but disabled for multimodal "
                    "audio distillation; this path can trigger checkpoint metadata mismatches."
                )
            elif hasattr(self.student, "gradient_checkpointing_enable"):
                self.student.gradient_checkpointing_enable()
                print("[info] gradient checkpointing enabled for student model")
            else:
                print("[warn] student model does not support gradient checkpointing")

        self.metrics_path = Path(cfg.data.output_dir) / "train_metrics.jsonl"
        self.swanlab = self._init_swanlab()
        self._audio_warning_count = 0
        self._audio_backend_hint_shown = False
        self._processor_audio_warning_shown = False
        self._sample_runtime_warning_count = 0
        self._generate_warning_count = 0
        self._generate_text_fallback_warning_count = 0

        # Cache for teacher generation (reuse for same clean audio)
        self._teacher_response_cache = {}
        self._audio_sanitize_warning_count = 0
        self._audio_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._audio_cache_size = 256
        self._eval_subset_indices: Optional[List[int]] = None
        self._eval_subset_cache_key: Optional[tuple[int, int, int]] = None
        self._last_val_metrics: Optional[Dict[str, float]] = None
        self._last_loss_skip_event: Optional[str] = None
        self._last_zero_loss_detail: Optional[Dict[str, Any]] = None
        try:
            self._student_forward_keys = set(inspect.signature(self.student.forward).parameters.keys())
        except Exception:
            self._student_forward_keys = {"input_ids", "attention_mask"}
        student_config = getattr(self.student, "config", None)
        student_model_type = str(getattr(student_config, "model_type", "")).lower()
        self._is_qwen_omni_runtime = "qwen2_5_omni" in student_model_type or ("qwen" in student_model_type and "omni" in student_model_type)
        # Use the legacy, verified generate() path for Qwen. The manual multimodal
        # path can collapse to empty/special-token responses and make eval accuracy
        # appear as zero even when validation samples are loaded correctly.
        self._manual_multimodal_generation = False
        self._manual_multimodal_generation_notice_shown = False

    def _clear_cuda_cache(self) -> None:
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[warn] cuda empty_cache failed; CUDA context may be invalid: {e}")
            try:
                torch.cuda.synchronize()
            except Exception as e:
                print(f"[warn] cuda synchronize failed; CUDA context may be invalid: {e}")
        gc.collect()

    def _is_oom_error(self, err: Exception) -> bool:
        text = str(err).lower()
        return "out of memory" in text or "cuda out of memory" in text

    def _is_fatal_cuda_error(self, err: Exception) -> bool:
        text = str(err).lower()
        fatal_keys = [
            "unspecified launch failure",
            "illegal memory access",
            "device-side assert",
            "device side assert",
            "cublas_status_execution_failed",
            "cudnn_status_execution_failed",
            "misaligned address",
            "launch timeout",
            "invalid configuration argument",
            "cuda error:",
        ]
        if not any(key in text for key in fatal_keys):
            return False
        nonfatal_keys = [
            "out of memory",
            "cuda out of memory",
        ]
        return not any(key in text for key in nonfatal_keys)

    def _is_audio_backend_error(self, err: Exception) -> bool:
        text = str(err).lower()
        keys = [
            "ffmpeg not found in path",
            "libtorchaudio.so",
            "could not load this library",
            "format not recognised",
            "file does not start with riff id",
        ]
        return any(k in text for k in keys)

    def _warn_skip_sample(self, sample: Dict[str, Any], err: Exception, reason: str) -> None:
        if self._sample_runtime_warning_count >= self.cfg.data.max_audio_load_warnings:
            return
        self._sample_runtime_warning_count += 1
        print(
            f"[warn] skip sample ({reason}): "
            f"noisy={sample.get('noisy_audio_path')} clean={sample.get('clean_audio_path')} err={err}"
        )

    def _ensure_finite(self, x: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.isfinite(x).all():
            raise SampleSkipError(f"non-finite tensor detected in {name}")
        return x

    def _sanitize_floating_tensor(self, value: torch.Tensor, name: str) -> torch.Tensor:
        if not torch.is_tensor(value) or not value.is_floating_point():
            return value
        if torch.isfinite(value).all():
            return value
        if self._sample_runtime_warning_count < self.cfg.data.max_audio_load_warnings:
            self._sample_runtime_warning_count += 1
            print(f"[warn] non-finite tensor sanitized: {name} shape={tuple(value.shape)}")
        sanitized = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        if not torch.isfinite(sanitized).all():
            raise SampleSkipError(f"failed to sanitize non-finite tensor in {name}")
        return sanitized

    def _sanitize_nested_floats(self, value: Any, name: str) -> Any:
        if torch.is_tensor(value):
            return self._sanitize_floating_tensor(value, name)
        if isinstance(value, list):
            return [self._sanitize_nested_floats(item, f"{name}[{idx}]") for idx, item in enumerate(value)]
        if isinstance(value, tuple):
            return tuple(self._sanitize_nested_floats(item, f"{name}[{idx}]") for idx, item in enumerate(value))
        if isinstance(value, dict):
            return {k: self._sanitize_nested_floats(v, f"{name}.{k}") for k, v in value.items()}
        return value

    def _sanitize_model_inputs(self, model_inputs: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        return {
            key: self._sanitize_nested_floats(value, f"{prefix}.{key}")
            for key, value in model_inputs.items()
        }

    def _normalize_reward_text(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value).strip().lower()
        if not text:
            return ""
        return re.sub(r"\s+", " ", text)

    def _normalize_reward_choices(self, choices: Optional[List[str]], target: str = "") -> List[str]:
        normalized: List[str] = []
        seen = set()
        for choice in choices or []:
            choice_norm = self._normalize_reward_text(choice)
            if not choice_norm or choice_norm in seen:
                continue
            seen.add(choice_norm)
            normalized.append(choice_norm)
        target_norm = self._normalize_reward_text(target)
        if target_norm and target_norm not in seen:
            normalized.append(target_norm)
        return normalized

    def _choice_match_is_negated(self, response_norm: str, start: int, choice_norm: str) -> bool:
        before = response_norm[max(0, start - 24):start]
        after_start = start + len(choice_norm)
        after = response_norm[after_start:after_start + 18]
        neg_before = re.search(
            r"(?:\bnot\b|\bno\b|\bwrong\b|\bn't\b|\binstead of\b|\brather than\b)\W*$",
            before,
        )
        neg_after = re.match(r"\W*(?:is\s+wrong|isn't|is\s+not|not)\b", after)
        return bool(neg_before or neg_after)

    def _select_choice_from_response(self, response: str, choices: Optional[List[str]], target: str = "") -> Optional[str]:
        response_norm = self._normalize_reward_text(response)
        choices_norm = self._normalize_reward_choices(choices, target=target)
        if not response_norm or not choices_norm:
            return None

        candidates: List[tuple[int, int, int, str]] = []
        cue_prefix = (
            r"(?:answer|option|choice|pick|choose|chose|selected|select|it's|it is|"
            r"i think(?: the answer is)?|maybe|probably|sounds like|looks like)"
        )
        for choice_norm in choices_norm:
            escaped = re.escape(choice_norm)
            patterns = [
                (0, rf"^\W*[\"']?{escaped}[\"']?\W*$"),
                (1, rf"(?<!\w){cue_prefix}\W+[\"']?{escaped}[\"']?(?!\w)"),
                (2, rf"(?<!\w)[\"']?{escaped}[\"']?(?!\w)"),
            ]
            for priority, pattern in patterns:
                matches = list(re.finditer(pattern, response_norm))
                if not matches:
                    continue
                for match in matches:
                    if self._choice_match_is_negated(response_norm, match.start(), choice_norm):
                        continue
                    candidates.append((priority, match.start(), -len(choice_norm), choice_norm))
                if candidates:
                    break

        if not candidates:
            return None
        candidates.sort()
        return candidates[0][3]

    def _response_contains_phrase(self, response: str, phrase: str) -> bool:
        response_norm = self._normalize_reward_text(response)
        phrase_norm = self._normalize_reward_text(phrase)
        if not response_norm or not phrase_norm:
            return False
        if response_norm == phrase_norm:
            return True
        if len(phrase_norm) <= 2:
            return bool(re.search(rf"(?<!\w){re.escape(phrase_norm)}(?!\w)", response_norm))
        if re.search(rf"(?<!\w){re.escape(phrase_norm)}(?!\w)", response_norm):
            return True
        return phrase_norm in response_norm

    def _compute_response_reward(
        self,
        response: str,
        *,
        target: str = "",
        teacher_response: str = "",
        choices: Optional[List[str]] = None,
        wrong_choice_reward: float = -1.0,
    ) -> float:
        response_norm = self._normalize_reward_text(response)
        target_norm = self._normalize_reward_text(target)
        teacher_norm = self._normalize_reward_text(teacher_response)
        choices_norm = self._normalize_reward_choices(choices, target=target_norm)

        selected_choice = self._select_choice_from_response(response_norm, choices_norm, target=target_norm)
        if selected_choice is not None:
            if target_norm and selected_choice == target_norm:
                return 1.0
            return float(wrong_choice_reward)

        if target_norm and self._response_contains_phrase(response_norm, target_norm):
            return 1.0

        if teacher_norm and self._response_contains_phrase(response_norm, teacher_norm):
            return 1.0

        response_words = set(response_norm.split())
        if target_norm:
            target_words = set(target_norm.split())
            overlap = target_words & response_words
            if overlap:
                return float(min(1.0, len(overlap) / max(1, len(target_words))))
        elif teacher_norm:
            teacher_words = set(teacher_norm.split())
            overlap = teacher_words & response_words
            if overlap:
                return float(min(1.0, len(overlap) / max(1, len(teacher_words))))

        return 0.0

    def _after_backward(self, step: int, batch: List[Dict[str, Any]], out: StepOutput) -> None:
        return

    def _after_optimizer_step(self, step: int) -> None:
        return

    def _get_audio_cached(self, path: str) -> torch.Tensor:
        cached = self._audio_cache.get(path)
        if cached is not None:
            self._audio_cache.move_to_end(path)
            return cached

        wav = load_audio_mono(path)
        if wav.ndim > 1:
            wav = wav.mean(dim=0)
        wav = wav.to(torch.float32).contiguous()
        if not torch.isfinite(wav).all():
            if self._audio_sanitize_warning_count < self.cfg.data.max_audio_load_warnings:
                self._audio_sanitize_warning_count += 1
                print(f"[warn] non-finite audio detected and sanitized: path={path}")
            wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        self._audio_cache[path] = wav
        self._audio_cache.move_to_end(path)
        while len(self._audio_cache) > self._audio_cache_size:
            self._audio_cache.popitem(last=False)
        return wav

    def _load_audio_parallel(self, paths: List[str]) -> Dict[str, Optional[torch.Tensor]]:
        """Load multiple audio files in parallel."""
        results: Dict[str, Optional[torch.Tensor]] = {}

        # First check cache
        uncached_paths = []
        for path in paths:
            cached = self._audio_cache.get(path)
            if cached is not None:
                self._audio_cache.move_to_end(path)
                results[path] = cached
            else:
                uncached_paths.append(path)

        if not uncached_paths:
            return results

        # Load uncached files in parallel
        num_workers = getattr(self.cfg.data, 'parallel_audio_load', 4)
        num_workers = max(1, min(num_workers, 8))

        def load_single(path: str):
            try:
                wav = load_audio_mono(path)
                if wav.ndim > 1:
                    wav = wav.mean(dim=0)
                wav = wav.to(torch.float32).contiguous()
                if not torch.isfinite(wav).all():
                    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
                return path, wav, None
            except Exception as e:
                return path, None, e

        if num_workers > 1 and len(uncached_paths) > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(load_single, p): p for p in uncached_paths}
                for future in as_completed(futures):
                    path, wav, err = future.result()
                    if wav is not None:
                        results[path] = wav
                        self._audio_cache[path] = wav
                        self._audio_cache.move_to_end(path)
                        while len(self._audio_cache) > self._audio_cache_size:
                            self._audio_cache.popitem(last=False)
                    else:
                        results[path] = None
        else:
            # Single-threaded fallback
            for path in uncached_paths:
                path, wav, err = load_single(path)
                if wav is not None:
                    results[path] = wav
                    self._audio_cache[path] = wav
                    self._audio_cache.move_to_end(path)
                    while len(self._audio_cache) > self._audio_cache_size:
                        self._audio_cache.popitem(last=False)
                else:
                    results[path] = None

        return results

    def _should_skip_loss(self, loss: torch.Tensor, epoch: int, batch_idx: int, step: int) -> bool:
        if not torch.isfinite(loss):
            self.optimizer.zero_grad(set_to_none=True)
            self._last_loss_skip_event = "nonfinite_loss_skip"
            rec = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "step": step,
                "event": "nonfinite_loss_skip",
            }
            self.log_metrics(rec)
            self._log_swanlab({"train/nonfinite_loss_skip": 1, "step": step})
            print(f"[warn] non-finite loss at step={step}, skip current batch")
            return True
        if abs(float(loss.item())) <= 1e-12:
            self._last_loss_skip_event = "zero_loss_skip"
            rec = {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "step": step,
                "event": "zero_loss_skip",
            }
            if self._last_zero_loss_detail:
                rec.update(self._last_zero_loss_detail)
            self.log_metrics(rec)
            self._log_swanlab({"train/zero_loss_skip": 1, "step": step})
            return True
        self._last_loss_skip_event = None
        return False

    def _init_swanlab(self):
        if not getattr(self.cfg, "swanlab", None) or not self.cfg.swanlab.enabled:
            return None
        try:
            swanlab = importlib.import_module("swanlab")

            swanlab.login()
            run_cfg = {
                "student_model": self.cfg.model.student_model,
                "teacher_model": self.cfg.model.teacher_model,
                "group_size": self.cfg.sdpo.group_size,
                "learning_rate": self.cfg.train.learning_rate,
                "batch_size": self.cfg.train.batch_size,
                "num_epochs": self.cfg.train.num_epochs,
                "policy_loss_weight": self.cfg.sdpo.policy_loss_weight,
                "distill_loss_weight": self.cfg.sdpo.distill_loss_weight,
            }
            swanlab.init(
                project=self.cfg.swanlab.project,
                name=self.cfg.swanlab.run_name or None,
                description=self.cfg.swanlab.description,
                config=run_cfg,
            )
            print("[info] swanlab initialized")
            return swanlab
        except Exception as e:
            print(f"[warn] swanlab disabled due to init error: {e}")
            return None

    def _log_swanlab(self, payload: Dict[str, Any]) -> None:
        if self.swanlab is None:
            return
        try:
            self.swanlab.log(payload)
        except Exception as e:
            print(f"[warn] swanlab log failed: {e}")

    def _build_inputs(
        self,
        text: str,
        wav: torch.Tensor,
        audio_path: str = "",
        target_device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build model inputs with audio using chat template approach (like spdal).

        Args:
            text: The text prompt
            wav: Audio tensor
            audio_path: Path to the audio file (used in message content for chat template)
        """
        target_device = target_device or self.student_device
        wav = wav.detach().cpu()
        wav_np = wav.numpy()

        # Debug: check if audio is being processed
        _debug_audio = False  # Disabled for performance

        if self.processor is not None and not self.cfg.model.force_text_only_omni:
            try:
                # Use chat template approach like spdal for Qwen2.5-Omni
                # Key difference from spdal: we use audio_path in message content (not wav_np)
                # and separately pass wav_np to the processor
                if hasattr(self.processor, "apply_chat_template"):
                    # Build message with audio path (like spdal)
                    # Use file path in message content, not numpy array
                    audio_for_message = audio_path if audio_path else wav_np
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": audio_for_message},
                                {"type": "text", "text": text},
                            ],
                        }
                    ]
                    # Apply chat template
                    text_formatted = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    # Process with audio numpy array (NOT the path)
                    enc = self.processor(
                        text=[text_formatted],
                        audio=[wav_np],
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                    )
                    # Check if audio features are present
                    has_audio_features = any(
                        k in enc
                        for k in [
                            "input_features",
                            "audio_values",
                            "audio_attention_mask",
                            "input_audio_embeds",
                        ]
                    )
                    if has_audio_features:
                        if _debug_audio:
                            self._debug_audio_count = getattr(self, '_debug_audio_count', 0) + 1
                            print(f"[debug] Audio input keys (chat template): {list(enc.keys())}")
                            for k, v in enc.items():
                                if hasattr(v, 'shape'):
                                    print(f"  {k}: {v.shape}")
                        return {k: v.to(target_device) for k, v in enc.items()}

                # Fallback: try original approach if chat template didn't work
                call_sig = inspect.signature(self.processor.__call__)
                params = call_sig.parameters
                base_kwargs: Dict[str, Any] = {"text": [text], "return_tensors": "pt", "padding": True}

                tried = False
                # Only try explicitly declared parameter names to avoid noisy "ignored kwarg" warnings.
                for audio_key in ["audio", "speech"]:
                    if audio_key not in params:
                        continue
                    for with_sr in [True, False]:
                        for audio_payload in ([wav_np], [wav]):
                            q = dict(base_kwargs)
                            if with_sr:
                                q["sampling_rate"] = 16000
                            q[audio_key] = audio_payload
                            try:
                                enc = self.processor(**q)
                            except TypeError:
                                continue
                            except Exception:
                                continue
                            tried = True
                            # Guard against processors that accept the kwarg but still drop audio silently.
                            has_audio_features = any(
                                k in enc
                                for k in [
                                    "input_features",
                                    "audio_values",
                                    "audio_attention_mask",
                                    "input_audio_embeds",
                                ]
                            )
                            if has_audio_features:
                                # Debug output for first few samples
                                if _debug_audio:
                                    self._debug_audio_count = getattr(self, '_debug_audio_count', 0) + 1
                                    pass  # Debug
                                    for k, v in enc.items():
                                        if hasattr(v, 'shape'):
                                            print(f"  {k}: {v.shape}")
                                return {k: v.to(target_device) for k, v in enc.items()}

                if tried and not self._processor_audio_warning_shown:
                    self._processor_audio_warning_shown = True
                    print("[warn] processor did not produce audio features; fallback to tokenizer-only inputs.")
                    # Debug: show what processor actually returned
                    if _debug_audio:
                        pass  # Debug
                if self._strict_audio_distillation:
                    raise SampleSkipError("processor did not produce usable audio features")
            except Exception:
                if self._strict_audio_distillation:
                    raise

        if self._strict_audio_distillation:
            raise SampleSkipError("processor unavailable for strict audio distillation")
        tok = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(target_device) for k, v in tok.items()}

    def _build_qwen_dialog_inputs(
        self,
        user_text: str,
        wav: torch.Tensor,
        audio_path: str = "",
        assistant_text: str = "",
        add_generation_prompt: Optional[bool] = None,
        target_device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Build Qwen Omni inputs with the answer placed in the assistant role."""
        target_device = target_device or self.student_device
        user_text = str(user_text or "")
        assistant_text = str(assistant_text or "")
        use_generation_prompt = (
            bool(add_generation_prompt)
            if add_generation_prompt is not None
            else not bool(assistant_text)
        )

        wav = wav.detach().cpu()
        wav_np = wav.numpy()

        if self.processor is not None and not self.cfg.model.force_text_only_omni:
            try:
                if hasattr(self.processor, "apply_chat_template"):
                    audio_for_message = audio_path if audio_path else wav_np
                    messages: List[Dict[str, Any]] = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "audio", "audio": audio_for_message},
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ]
                    if assistant_text:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": [{"type": "text", "text": assistant_text}],
                            }
                        )
                    formatted_text = self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=use_generation_prompt,
                    )
                    enc = self.processor(
                        text=[formatted_text],
                        audio=[wav_np],
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                    )
                    has_audio_features = any(
                        k in enc
                        for k in [
                            "input_features",
                            "audio_values",
                            "audio_attention_mask",
                            "input_audio_embeds",
                        ]
                    )
                    if has_audio_features:
                        return {k: v.to(target_device) for k, v in enc.items()}

                if self._strict_audio_distillation:
                    raise SampleSkipError("processor did not produce usable Qwen dialog audio features")
            except Exception:
                if self._strict_audio_distillation:
                    raise

        if self._strict_audio_distillation:
            raise SampleSkipError("processor unavailable for strict Qwen dialog distillation")
        raw_text = user_text + assistant_text
        tok = self.tokenizer([raw_text], return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(target_device) for k, v in tok.items()}

    def _build_inputs_batch(
        self,
        texts: List[str],
        wav: torch.Tensor,
        audio_path: str = "",
        target_device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        target_device = target_device or self.student_device
        if not texts:
            return {}

        wav = wav.detach().cpu()
        wav_np = wav.numpy()

        if self.processor is not None and not self.cfg.model.force_text_only_omni:
            try:
                if hasattr(self.processor, "apply_chat_template"):
                    audio_for_message = audio_path if audio_path else wav_np
                    formatted_texts: List[str] = []
                    for text in texts:
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "audio", "audio": audio_for_message},
                                    {"type": "text", "text": text},
                                ],
                            }
                        ]
                        formatted_texts.append(
                            self.processor.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                        )
                    enc = self.processor(
                        text=formatted_texts,
                        audio=[wav_np] * len(formatted_texts),
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                    )
                    has_audio_features = any(
                        k in enc
                        for k in [
                            "input_features",
                            "audio_values",
                            "audio_attention_mask",
                            "input_audio_embeds",
                        ]
                    )
                    if has_audio_features:
                        return {k: v.to(target_device) for k, v in enc.items()}

                call_sig = inspect.signature(self.processor.__call__)
                params = call_sig.parameters
                base_kwargs: Dict[str, Any] = {"text": texts, "return_tensors": "pt", "padding": True}
                tried = False
                for audio_key in ["audio", "speech"]:
                    if audio_key not in params:
                        continue
                    for with_sr in [True, False]:
                        for audio_payload in ([wav_np] * len(texts), [wav] * len(texts)):
                            q = dict(base_kwargs)
                            if with_sr:
                                q["sampling_rate"] = 16000
                            q[audio_key] = audio_payload
                            try:
                                enc = self.processor(**q)
                            except TypeError:
                                continue
                            except Exception:
                                continue
                            tried = True
                            has_audio_features = any(
                                k in enc
                                for k in [
                                    "input_features",
                                    "audio_values",
                                    "audio_attention_mask",
                                    "input_audio_embeds",
                                ]
                            )
                            if has_audio_features:
                                return {k: v.to(target_device) for k, v in enc.items()}

                if tried and not self._processor_audio_warning_shown:
                    self._processor_audio_warning_shown = True
                    print("[warn] processor did not produce audio features; fallback to tokenizer-only inputs.")
                if self._strict_audio_distillation:
                    raise SampleSkipError("processor did not produce usable audio features")
            except Exception:
                if self._strict_audio_distillation:
                    raise

        if self._strict_audio_distillation:
            raise SampleSkipError("processor unavailable for strict audio distillation")
        tok = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        return {k: v.to(target_device) for k, v in tok.items()}

    def _mask_response(self, ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
        m = torch.zeros_like(ids, dtype=torch.bool)
        if prompt_len < ids.shape[1]:
            m[:, prompt_len:] = True
        return m

    def _avg_logprob_from_ids(self, model: Any, ids: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=ids)
        logits = out.logits[:, :-1, :]
        labels = ids[:, 1:].to(logits.device)
        mask = response_mask[:, 1:].to(logits.device)
        logp = torch.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask).sum() / mask.sum().clamp(min=1)

    def _decode_response(self, seq: torch.Tensor, prompt_len: int) -> str:
        resp_ids = seq[:, prompt_len:]
        if resp_ids.numel() == 0:
            return ""
        return self.tokenizer.decode(resp_ids[0], skip_special_tokens=True)

    def _normalize_eval_text(self, text: Any) -> str:
        if text is None:
            return ""
        return " ".join(str(text).strip().lower().split())

    def _contains_phrase(self, text: str, phrase: str) -> bool:
        if not text or not phrase:
            return False
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        return re.search(pattern, text) is not None

    def _get_eval_target(self, sample: Dict[str, Any]) -> str:
        meta = sample.get("meta", {}) or {}
        return self._normalize_eval_text(
            sample.get("target")
            or meta.get("target")
            or meta.get("ground_truth")
            or meta.get("answer")
        )

    def _get_eval_candidates(self, sample: Dict[str, Any]) -> List[str]:
        meta = sample.get("meta", {}) or {}
        raw_candidates: List[Any] = []
        raw_candidates.extend(meta.get("choices", []) or [])
        raw_candidates.extend(
            [
                sample.get("target"),
                meta.get("target"),
                meta.get("ground_truth"),
                meta.get("answer"),
            ]
        )
        deduped: List[str] = []
        seen = set()
        for item in raw_candidates:
            norm = self._normalize_eval_text(item)
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(norm)
        deduped.sort(key=len, reverse=True)
        return deduped

    def _canonicalize_eval_response(self, response: Any, sample: Dict[str, Any]) -> str:
        response_norm = self._normalize_eval_text(response)
        if not response_norm:
            return ""
        for candidate in self._get_eval_candidates(sample):
            if response_norm == candidate or self._contains_phrase(response_norm, candidate):
                return candidate
        return response_norm

    def _get_eval_primary_group(self, sample: Dict[str, Any]) -> str:
        meta = sample.get("meta", {}) or {}
        for key in ("noise_type", "type", "task_type", "question_type", "category"):
            value = meta.get(key, sample.get(key))
            norm = self._normalize_eval_text(value)
            if norm:
                return norm
        return "__all__"

    def _get_eval_secondary_group(self, sample: Dict[str, Any]) -> str:
        meta = sample.get("meta", {}) or {}
        for key in ("snr", "difficulty", "subset"):
            value = meta.get(key, sample.get(key))
            norm = self._normalize_eval_text(value)
            if norm:
                return norm
        return "__default__"

    def _build_balanced_eval_subset_indices(self, dataset: Any, subset_size: int) -> List[int]:
        dataset_len = len(dataset)
        if subset_size <= 0 or dataset_len <= subset_size:
            return list(range(dataset_len))

        rng = random.Random(self.cfg.train.seed)
        grouped: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
        for idx in range(dataset_len):
            sample = dataset[idx]
            primary = self._get_eval_primary_group(sample)
            secondary = self._get_eval_secondary_group(sample)
            grouped[primary][secondary].append(idx)

        primary_queues: Dict[str, deque[int]] = {}
        for primary, secondary_groups in grouped.items():
            secondary_keys = list(secondary_groups.keys())
            rng.shuffle(secondary_keys)
            secondary_queues: Dict[str, deque[int]] = {}
            for secondary in secondary_keys:
                indices = list(secondary_groups[secondary])
                rng.shuffle(indices)
                secondary_queues[secondary] = deque(indices)

            interleaved: List[int] = []
            while True:
                progressed = False
                for secondary in secondary_keys:
                    queue = secondary_queues[secondary]
                    if not queue:
                        continue
                    interleaved.append(queue.popleft())
                    progressed = True
                if not progressed:
                    break
            primary_queues[primary] = deque(interleaved)

        primary_keys = list(primary_queues.keys())
        rng.shuffle(primary_keys)

        selected: List[int] = []
        active_keys = [key for key in primary_keys if primary_queues[key]]
        while active_keys and len(selected) < subset_size:
            next_active: List[str] = []
            for key in active_keys:
                queue = primary_queues[key]
                if not queue:
                    continue
                selected.append(queue.popleft())
                if queue:
                    next_active.append(key)
                if len(selected) >= subset_size:
                    break
            active_keys = next_active

        if len(selected) < subset_size:
            leftovers: List[int] = []
            for key in primary_keys:
                leftovers.extend(list(primary_queues[key]))
            selected.extend(leftovers[: subset_size - len(selected)])

        return sorted(selected)

    def _response_is_correct(self, response: str, sample: Dict[str, Any]) -> bool:
        target = self._get_eval_target(sample)
        if not target:
            return False
        return self._canonicalize_eval_response(response, sample) == target

    def _score_response_with_audio(
        self,
        model: Any,
        prompt: str,
        response: str,
        wav: torch.Tensor,
        audio_path: str = "",
        target_device: Optional[torch.device] = None,
        prompt_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_device = target_device or self.student_device
        if self._is_qwen_omni_runtime:
            model_inp = self._build_qwen_dialog_inputs(
                prompt,
                wav,
                audio_path=audio_path,
                assistant_text=response,
                add_generation_prompt=False,
                target_device=target_device,
            )
            if prompt_ids is None:
                prompt_inp = self._build_qwen_dialog_inputs(
                    prompt,
                    wav,
                    audio_path=audio_path,
                    assistant_text="",
                    add_generation_prompt=True,
                    target_device=target_device,
                )
                prompt_ids = prompt_inp.get("input_ids")
        else:
            full_text = prompt + response
            model_inp = self._build_inputs(
                full_text,
                wav,
                audio_path=audio_path,
                target_device=target_device,
            )
            prompt_inp = self._build_inputs(
                prompt,
                wav,
                audio_path=audio_path,
                target_device=target_device,
            )
            prompt_ids = prompt_inp.get("input_ids")

        full_ids = model_inp.get("input_ids")
        if full_ids is None:
            full_text = prompt + response
            full_ids = self.tokenizer(full_text, return_tensors="pt", truncation=True).input_ids.to(target_device)
            model_inp = {"input_ids": full_ids}

        if prompt_ids is None:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(target_device)
        prompt_len = int(min(prompt_ids.shape[1], full_ids.shape[1]))
        response_mask = self._mask_response(full_ids, prompt_len)

        safe_model_inp = self._sanitize_model_inputs(model_inp, "score.single_inputs")
        out = self._teacher_forward_eval_mode(safe_model_inp) if model is self.teacher else model(**safe_model_inp)
        logits = self._sanitize_floating_tensor(out.logits[:, :-1, :], "score.single_logits")
        labels = full_ids[:, 1:]
        mask = response_mask[:, 1:]
        labels = labels.to(logits.device)
        mask = mask.to(logits.device)
        logp = torch.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask).sum() / mask.sum().clamp(min=1)

    def _score_responses_batch(
        self,
        model: Any,
        prompt: str,
        responses: List[str],
        wav: torch.Tensor,
        audio_path: str = "",
        target_device: Optional[torch.device] = None,
        prompt_ids: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Batch score multiple responses with a single forward pass.

        This is much faster than scoring each response individually.
        """
        if not responses:
            return []
        target_device = target_device or self.student_device
        use_multimodal = wav is not None and self.processor is not None and not self.cfg.model.force_text_only_omni

        if use_multimodal and self._is_qwen_omni_runtime:
            if not getattr(self, "_qwen_omni_sequential_score_notice_shown", False):
                self._qwen_omni_sequential_score_notice_shown = True
                print(
                    "[info] Qwen2.5-Omni sequential multimodal scoring enabled to "
                    "avoid batched audio-tower instability."
                )
            qwen_prompt_ids = prompt_ids
            if qwen_prompt_ids is None:
                prompt_inp = self._build_qwen_dialog_inputs(
                    prompt,
                    wav,
                    audio_path=audio_path,
                    assistant_text="",
                    add_generation_prompt=True,
                    target_device=target_device,
                )
                qwen_prompt_ids = prompt_inp.get("input_ids")
            results: List[torch.Tensor] = []
            for response in responses:
                results.append(
                    self._score_response_with_audio(
                        model,
                        prompt,
                        response,
                        wav,
                        audio_path=audio_path,
                        target_device=target_device,
                        prompt_ids=qwen_prompt_ids,
                    )
            )
            return results

        full_texts = [prompt + resp for resp in responses]
        if use_multimodal:
            model_inp = self._build_inputs_batch(
                full_texts,
                wav,
                audio_path=audio_path,
                target_device=target_device,
            )
            prompt_inp = self._build_inputs_batch(
                [prompt],
                wav,
                audio_path=audio_path,
                target_device=target_device,
            )
            full_ids = model_inp.get("input_ids")
            prompt_ids = prompt_inp.get("input_ids")
        else:
            model_inp = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(target_device)
            full_ids = model_inp.get("input_ids")
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(target_device)

        if full_ids is None:
            full_ids = self.tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(target_device)
            model_inp = {"input_ids": full_ids}
        if prompt_ids is None:
            prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(target_device)

        prompt_len = int(min(prompt_ids.shape[1], full_ids.shape[1]))
        attention_mask = model_inp.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(full_ids, device=target_device)

        safe_model_inp = self._sanitize_model_inputs(model_inp, "score.batch_inputs")
        out = self._teacher_forward_eval_mode(safe_model_inp) if model is self.teacher else model(**safe_model_inp)
        logits = self._sanitize_floating_tensor(out.logits[:, :-1, :], "score.batch_logits")  # (batch, seq_len-1, vocab)
        labels = full_ids[:, 1:].to(logits.device)  # (batch, seq_len-1)

        # Create masks for each response
        batch_size = full_ids.shape[0]
        masks = []
        for i in range(batch_size):
            seq_len = int((attention_mask[i] == 1).sum().item())
            resp_start = min(prompt_len, seq_len)
            mask = torch.zeros_like(full_ids[i, 1:], dtype=torch.bool)
            if resp_start < seq_len:
                mask[resp_start:] = True
            masks.append(mask)
        response_mask = torch.stack(masks).to(logits.device)

        # Compute log probs
        logp = torch.log_softmax(logits, dim=-1)
        token_lp = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        # Compute avg logprob for each response
        results = []
        for i in range(batch_size):
            mask = response_mask[i]
            lp = token_lp[i]
            avg_lp = (lp * mask).sum() / mask.sum().clamp(min=1)
            results.append(avg_lp)

        return results

    def _sanitize_generation_inputs(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize inputs for generation - same as spdal to avoid next_sequence_length bug."""
        generation_exclude_keys = {
            "next_sequence_length",
            "num_logits_to_keep",
            "logits_to_keep",
            "cache_position",
            "past_key_values",
        }
        # Keys to keep for generation (including audio-related keys)
        keep_keys = {"input_ids", "attention_mask", "feature_attention_mask", "input_features"}
        sanitized: Dict[str, Any] = {}
        for key, value in model_inputs.items():
            if key in generation_exclude_keys:
                continue
            if key in self._student_forward_keys or key in keep_keys:
                sanitized[key] = value
        return sanitized

    def _clone_model_inputs(self, model_inputs: Dict[str, Any]) -> Dict[str, Any]:
        cloned: Dict[str, Any] = {}
        for key, value in model_inputs.items():
            if torch.is_tensor(value):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = [item.clone() if torch.is_tensor(item) else item for item in value]
            elif isinstance(value, tuple):
                cloned[key] = tuple(item.clone() if torch.is_tensor(item) else item for item in value)
            elif isinstance(value, dict):
                cloned[key] = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in value.items()}
            else:
                cloned[key] = value
        return self._sanitize_model_inputs(cloned, "cloned_inputs")

    def _append_generation_token(self, model_inputs: Dict[str, Any], token_id: int) -> Dict[str, Any]:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise SampleSkipError("input_ids missing during manual qwen generation")
        next_token = torch.tensor([[int(token_id)]], device=input_ids.device, dtype=input_ids.dtype)
        model_inputs["input_ids"] = torch.cat([input_ids, next_token], dim=1)
        attention_mask = model_inputs.get("attention_mask")
        if torch.is_tensor(attention_mask):
            next_mask = torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)
            model_inputs["attention_mask"] = torch.cat([attention_mask, next_mask], dim=1)
        return model_inputs

    def _sample_next_token(
        self,
        next_token_logits: torch.Tensor,
        *,
        sampling: bool,
        temperature: float,
        top_p: float,
        blocked_token_ids: Optional[set[int]] = None,
    ) -> int:
        logits = self._sanitize_floating_tensor(next_token_logits.float(), "generation.logits")
        if blocked_token_ids:
            blocked = [tok for tok in blocked_token_ids if 0 <= int(tok) < logits.shape[-1]]
            if blocked:
                logits[..., blocked] = float("-inf")
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if not sampling:
            return int(torch.argmax(logits, dim=-1).item())
        temp = float(max(temperature, 1e-5))
        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        probs = self._sanitize_floating_tensor(probs, "generation.probs")
        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative_probs > float(top_p)
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            sorted_probs = self._sanitize_floating_tensor(sorted_probs, "generation.top_p_probs")
            sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, sampled_idx)
            return int(next_token.item())
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        probs = self._sanitize_floating_tensor(probs, "generation.renorm_probs")
        next_token = torch.multinomial(probs, num_samples=1)
        return int(next_token.item())

    @torch.no_grad()
    def _manual_generate_sequence(
        self,
        model: Any,
        model_inputs: Dict[str, Any],
        *,
        sampling: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> torch.Tensor:
        prepared_inputs = self._sanitize_model_inputs(self._clone_model_inputs(model_inputs), "manual_generation")
        start_input_ids = prepared_inputs.get("input_ids")
        if start_input_ids is None:
            raise SampleSkipError("input_ids missing for manual qwen generation")
        target_max_new_tokens = max(1, int(max_new_tokens))
        terminators = {
            int(tok)
            for tok in (
                getattr(self.tokenizer, "eos_token_id", None),
                getattr(self.tokenizer, "pad_token_id", None),
            )
            if tok is not None and int(tok) >= 0
        }
        blocked_first_tokens = set(terminators)
        for tok in getattr(self.tokenizer, "all_special_ids", []) or []:
            try:
                tok_id = int(tok)
            except Exception:
                continue
            if tok_id >= 0:
                blocked_first_tokens.add(tok_id)
        prev_training = bool(getattr(model, "training", False))
        if prev_training:
            model.eval()
        try:
            for token_idx in range(target_max_new_tokens):
                prepared_inputs = self._sanitize_model_inputs(prepared_inputs, "manual_generation.step")
                outputs = model(**prepared_inputs)
                logits = self._sanitize_floating_tensor(outputs.logits[:, -1, :], "manual_generation.output_logits")
                next_token_id = self._sample_next_token(
                    logits,
                    sampling=sampling,
                    temperature=temperature,
                    top_p=top_p,
                    blocked_token_ids=blocked_first_tokens if token_idx == 0 else None,
                )
                if token_idx > 0 and next_token_id in terminators:
                    break
                if token_idx == 0 and next_token_id in terminators:
                    alt_logits = logits.clone()
                    for tok in blocked_first_tokens:
                        if 0 <= int(tok) < alt_logits.shape[-1]:
                            alt_logits[..., int(tok)] = float("-inf")
                    next_token_id = int(torch.argmax(alt_logits, dim=-1).item())
                prepared_inputs = self._append_generation_token(prepared_inputs, next_token_id)
                if token_idx == 0:
                    blocked_first_tokens = set()
                if next_token_id in terminators:
                    break
        finally:
            if prev_training:
                model.train()

        generated_ids = prepared_inputs.get("input_ids")
        if generated_ids is None:
            return start_input_ids.clone()
        return generated_ids

    def _maybe_log_manual_multimodal_generation(self) -> None:
        if self._manual_multimodal_generation_notice_shown:
            return
        self._manual_multimodal_generation_notice_shown = True
        print("[info] Qwen manual multimodal generation enabled to reduce peak VRAM.")

    @torch.no_grad()
    def _manual_generate_response(
        self,
        model: Any,
        model_inputs: Dict[str, Any],
        *,
        sampling: bool,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        decode_prompt_len: int,
    ) -> str:
        generated_ids = self._manual_generate_sequence(
            model,
            model_inputs,
            sampling=sampling,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        if generated_ids.ndim != 2 or generated_ids.shape[1] <= decode_prompt_len:
            return ""
        return self.tokenizer.decode(generated_ids[0, decode_prompt_len:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def _sample_candidates(self, student_inputs: Dict[str, torch.Tensor], prompt_text: str = "") -> List[torch.Tensor]:
        # Use the same approach as spdal - sanitize inputs before generation
        # This removes problematic keys like next_sequence_length that cause conflicts
        sanitized_inputs = self._sanitize_generation_inputs(student_inputs)

        # Keep input_ids, attention_mask, and audio features for generation
        gen_inputs: Dict[str, Any] = {}
        if "input_ids" in sanitized_inputs:
            gen_inputs["input_ids"] = sanitized_inputs["input_ids"].clone()
        if "attention_mask" in sanitized_inputs:
            gen_inputs["attention_mask"] = sanitized_inputs["attention_mask"].clone()
        # Pass audio features to generation (critical for multimodal generation)
        if "feature_attention_mask" in sanitized_inputs:
            gen_inputs["feature_attention_mask"] = sanitized_inputs["feature_attention_mask"].clone()
        if "input_features" in sanitized_inputs:
            # Ensure input_features has the correct shape for generation
            input_features = sanitized_inputs["input_features"]
            # Qwen2.5-Omni expects: (batch, feature_seq, hidden)
            if input_features.dim() == 3:
                gen_inputs["input_features"] = input_features
        if not gen_inputs:
            raise SampleSkipError("no input_ids available for generation")

        # Temporarily disable gradient checkpointing for generation
        # Gradient checkpointing is incompatible with use_cache=True in generation
        gc_was_enabled = False
        if hasattr(self.student, "gradient_checkpointing_disable"):
            if getattr(self.student, "gradient_checkpointing", False):
                gc_was_enabled = True
                self.student.gradient_checkpointing_disable()

        if self._manual_multimodal_generation:
            try:
                self._maybe_log_manual_multimodal_generation()
                prompt_ids = gen_inputs.get("input_ids")
                if prompt_ids is None:
                    raise SampleSkipError("input_ids missing for manual candidate generation")
                prompt_len = int(prompt_ids.shape[1])
                seqs: List[torch.Tensor] = []
                attempts = 0
                num_candidates = max(1, int(self.cfg.sdpo.group_size))
                max_attempts = max(num_candidates * 3, num_candidates + 2)
                last_err: Optional[Exception] = None
                while len(seqs) < num_candidates and attempts < max_attempts:
                    attempts += 1
                    try:
                        seq = self._manual_generate_sequence(
                            self.student,
                            gen_inputs,
                            sampling=True,
                            temperature=float(self.cfg.sdpo.temperature),
                            top_p=float(self.cfg.sdpo.top_p),
                            max_new_tokens=int(self.cfg.sdpo.max_new_tokens),
                        )
                    except Exception as e:
                        if self._is_fatal_cuda_error(e):
                            raise
                        last_err = e
                        continue
                    if seq.ndim != 2 or seq.shape[1] <= prompt_len:
                        continue
                    response = self.tokenizer.decode(seq[0, prompt_len:], skip_special_tokens=True).strip()
                    if response:
                        seqs.append(seq)
                if not seqs:
                    if self._generate_warning_count < self.cfg.data.max_audio_load_warnings:
                        self._generate_warning_count += 1
                        detail = f": {last_err}" if last_err is not None else ""
                        print(f"[warn] manual candidate generation produced no valid text; fallback to generate(){detail}")
                else:
                    return seqs
            finally:
                if gc_was_enabled and hasattr(self.student, "gradient_checkpointing_enable"):
                    self.student.gradient_checkpointing_enable()

        gen_kwargs = {
            "max_new_tokens": self.cfg.sdpo.max_new_tokens,
            "do_sample": True,
            "temperature": self.cfg.sdpo.temperature,
            "top_p": self.cfg.sdpo.top_p,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            # Must disable caching for QLoRA + audio features compatibility
            "use_cache": False,
        }

        # Also set on generation_config to ensure it takes effect
        if hasattr(self.student, "generation_config") and self.student.generation_config is not None:
            self.student.generation_config.use_cache = False

        seqs: List[torch.Tensor] = []
        num_candidates = max(1, int(self.cfg.sdpo.group_size))

        try:
            gen_inputs = self._sanitize_model_inputs(gen_inputs, "student.generate_inputs")
            out = self.student.generate(
                **gen_inputs,
                **{**gen_kwargs, "num_return_sequences": num_candidates}
            )
            if out.ndim == 2:
                for i in range(out.shape[0]):
                    seqs.append(out[i:i+1])
            else:
                seqs.append(out.unsqueeze(0))
        except Exception as e:
            if self._is_fatal_cuda_error(e):
                raise
            err_str = str(e)
            if self._generate_warning_count < self.cfg.data.max_audio_load_warnings:
                self._generate_warning_count += 1
                print(f"[warn] candidate generate failed: {err_str}")

            # Check if it's the known bug - if so, try text-only tokenizer approach
            if "next_sequence_length" in err_str or "Key and Value must have the same sequence length" in err_str:
                try:
                    # Fallback: use raw tokenizer instead of model inputs
                    text_inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=gen_inputs["input_ids"].shape[1]
                    )
                    text_inputs = {k: v.to(self.student_device) for k, v in text_inputs.items()}
                    text_inputs = self._sanitize_model_inputs(text_inputs, "student.generate_text_fallback")
                    out = self.student.generate(
                        input_ids=text_inputs["input_ids"],
                        attention_mask=text_inputs.get("attention_mask"),
                        **{**gen_kwargs, "num_return_sequences": num_candidates}
                    )
                    if out.ndim == 2:
                        for i in range(out.shape[0]):
                            seqs.append(out[i:i+1])
                    else:
                        seqs.append(out.unsqueeze(0))
                    print("[info] Fallback to text-only generation succeeded")
                except Exception as fallback_err:
                    if self._is_fatal_cuda_error(fallback_err):
                        raise
                    print(f"[warn] fallback generate also failed: {fallback_err}")

            if not seqs:
                raise SampleSkipError(f"generate failed: {e}") from e
        finally:
            # Re-enable gradient checkpointing even when generation fails.
            if gc_was_enabled and hasattr(self.student, "gradient_checkpointing_enable"):
                self.student.gradient_checkpointing_enable()

        if not seqs:
            raise SampleSkipError("no valid candidates generated")
        return seqs

    def _generate_teacher_response(
        self,
        teacher_inputs: Dict[str, torch.Tensor],
        prompt_len: int
    ) -> str:
        """Generate a single teacher response (greedy decoding for speed)."""
        sanitized_inputs = self._sanitize_generation_inputs(teacher_inputs)

        gen_inputs: Dict[str, Any] = {}
        if "input_ids" in sanitized_inputs:
            gen_inputs["input_ids"] = sanitized_inputs["input_ids"].clone()
        if "attention_mask" in sanitized_inputs:
            gen_inputs["attention_mask"] = sanitized_inputs["attention_mask"].clone()
        if "feature_attention_mask" in sanitized_inputs:
            gen_inputs["feature_attention_mask"] = sanitized_inputs["feature_attention_mask"].clone()
        if "input_features" in sanitized_inputs:
            gen_inputs["input_features"] = sanitized_inputs["input_features"]

        if not gen_inputs:
            return ""

        if self._manual_multimodal_generation:
            self._maybe_log_manual_multimodal_generation()
            response = self._manual_generate_response(
                self.teacher,
                gen_inputs,
                sampling=False,
                temperature=0.5,
                top_p=1.0,
                max_new_tokens=int(self.cfg.sdpo.max_new_tokens),
                decode_prompt_len=prompt_len,
            )
            if response:
                return response

        # Use greedy decoding for teacher (faster than sampling)
        gen_kwargs = {
            "max_new_tokens": self.cfg.sdpo.max_new_tokens,
            "do_sample": False,  # Greedy for speed
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "remove_invalid_values": True,
            "use_cache": True,  # Teacher can use cache (no training)
        }

        try:
            gen_inputs = self._sanitize_model_inputs(gen_inputs, "teacher.generate_inputs")
            out = self.teacher.generate(**gen_inputs, **gen_kwargs)
            if out.ndim > 1:
                out = out[0]
            return self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        except Exception as e:
            if self._is_fatal_cuda_error(e):
                raise
            return ""

    @torch.no_grad()
    def _generate_student_response(
        self,
        student_inputs: Dict[str, torch.Tensor],
        prompt_len: int,
    ) -> str:
        sanitized_inputs = self._sanitize_generation_inputs(student_inputs)

        gen_inputs: Dict[str, Any] = {}
        if "input_ids" in sanitized_inputs:
            gen_inputs["input_ids"] = sanitized_inputs["input_ids"].clone()
        if "attention_mask" in sanitized_inputs:
            gen_inputs["attention_mask"] = sanitized_inputs["attention_mask"].clone()
        if "feature_attention_mask" in sanitized_inputs:
            gen_inputs["feature_attention_mask"] = sanitized_inputs["feature_attention_mask"].clone()
        if "input_features" in sanitized_inputs:
            input_features = sanitized_inputs["input_features"]
            if input_features.dim() == 3:
                gen_inputs["input_features"] = input_features.clone()

        if not gen_inputs:
            return ""

        if self._manual_multimodal_generation:
            self._maybe_log_manual_multimodal_generation()
            response = self._manual_generate_response(
                self.student,
                gen_inputs,
                sampling=False,
                temperature=0.5,
                top_p=1.0,
                max_new_tokens=int(self.cfg.sdpo.max_new_tokens),
                decode_prompt_len=prompt_len,
            )
            if response:
                return response

        gen_kwargs = {
            "max_new_tokens": self.cfg.sdpo.max_new_tokens,
            "do_sample": False,
            "num_return_sequences": 1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": False,
        }

        if hasattr(self.student, "generation_config") and self.student.generation_config is not None:
            self.student.generation_config.use_cache = False

        out = self.student.generate(**gen_inputs, **gen_kwargs)
        if out.ndim > 1:
            out = out[0]
        return self.tokenizer.decode(out[prompt_len:], skip_special_tokens=True).strip()

    def _teacher_forward_eval_mode(self, model_inputs: Dict[str, torch.Tensor]):
        shared = self.teacher is self.student
        prev = None
        if shared:
            prev = self.teacher.training
            self.teacher.eval()
        # Use no_grad for teacher forward (inference_mode causes issues with autograd)
        with torch.no_grad():
            safe_inputs = self._sanitize_model_inputs(model_inputs, "teacher.forward_inputs")
            out = self.teacher(**safe_inputs)
            if hasattr(out, "logits") and torch.is_tensor(out.logits):
                out.logits = self._sanitize_floating_tensor(out.logits, "teacher.forward_logits")
        if shared and prev:
            self.teacher.train()
        return out

    def compute_step(self, batch: List[Dict[str, Any]]) -> StepOutput:
        policy_list = []
        distill_list = []
        reward_list = []
        skipped_samples = 0
        valid_samples = 0

        # Pre-load all audio files in parallel for the batch
        all_audio_paths = []
        for idx, sample in enumerate(batch):
            noisy_path = sample.get("noisy_audio_path", "")
            clean_path = sample.get("clean_audio_path", "")
            if noisy_path and clean_path:
                all_audio_paths.extend([noisy_path, clean_path])

        # Load audio in parallel
        audio_results = {}
        if all_audio_paths:
            results = self._load_audio_parallel(all_audio_paths)
            audio_results = {path: results.get(path) for path in all_audio_paths}

        # Group samples by clean_path: one teacher generation per clean audio
        clean_path_to_samples = {}
        for idx, sample in enumerate(batch):
            clean_path = sample.get("clean_audio_path", "")
            if clean_path:
                if clean_path not in clean_path_to_samples:
                    clean_path_to_samples[clean_path] = []
                clean_path_to_samples[clean_path].append((idx, sample))

        # Pre-generate teacher responses for each unique clean audio
        clean_path_to_teacher_response = {}
        clean_path_to_teacher_prompt_ids = {}
        clean_path_to_teacher_prompt = {}
        for clean_path, samples in clean_path_to_samples.items():
            # Get the first sample to get prompt
            first_sample = samples[0][1]
            prompt = first_sample["prompt"]

            clean = audio_results.get(clean_path)
            if clean is None:
                continue

            # Check cache first
            clean_audio_hash = hash(clean.cpu().numpy()[:100].tobytes()) if hasattr(clean, 'shape') else None
            if clean_audio_hash and clean_audio_hash in self._teacher_response_cache:
                teacher_response = self._teacher_response_cache[clean_audio_hash]
            else:
                # Generate teacher response for this clean audio
                try:
                    teacher_inp = self._build_inputs(
                        prompt,
                        clean,
                        audio_path=clean_path,
                        target_device=self.teacher_device,
                    )
                    prompt_ids = teacher_inp.get("input_ids")
                    if prompt_ids is not None:
                        clean_path_to_teacher_prompt_ids[clean_path] = prompt_ids
                        clean_path_to_teacher_prompt[clean_path] = prompt
                        prompt_len = int(prompt_ids.shape[1])
                        teacher_response = self._generate_teacher_response(teacher_inp, prompt_len)
                        if clean_audio_hash and teacher_response:
                            self._teacher_response_cache[clean_audio_hash] = teacher_response
                    else:
                        teacher_response = ""
                except Exception as e:
                    if self._is_fatal_cuda_error(e):
                        raise
                    teacher_response = ""

            teacher_response = teacher_response.strip() if isinstance(teacher_response, str) else teacher_response
            if teacher_response:
                clean_path_to_teacher_response[clean_path] = teacher_response

        # Now process each sample
        for sample in batch:
            prompt = sample["prompt"]
            target = sample.get("target", "")

            noisy_path = sample.get("noisy_audio_path", "")
            clean_path = sample.get("clean_audio_path", "")

            try:
                noisy = audio_results.get(noisy_path)
                clean = audio_results.get(clean_path)

                # Handle failed loads
                if noisy is None or clean is None:
                    err_msg = f"audio load failed: noisy={noisy_path} clean={clean_path}"
                    should_skip = self.cfg.data.skip_bad_audio
                    if should_skip:
                        skipped_samples += 1
                        if self._audio_warning_count < self.cfg.data.max_audio_load_warnings:
                            self._audio_warning_count += 1
                            print("[warn] skip bad audio sample: " + err_msg)
                        continue
                    raise RuntimeError(err_msg)

                # Type assertion for type checker (noisy and clean are guaranteed non-None here)
                assert noisy is not None and clean is not None

            except Exception as e:
                should_skip = self.cfg.data.skip_bad_audio or self._is_audio_backend_error(e)
                if should_skip:
                    skipped_samples += 1
                    if self._audio_warning_count < self.cfg.data.max_audio_load_warnings:
                        self._audio_warning_count += 1
                        print(
                            "[warn] skip bad audio sample: "
                            f"noisy={noisy_path} clean={clean_path} err={e}"
                        )
                    if self._is_audio_backend_error(e) and not self._audio_backend_hint_shown:
                        self._audio_backend_hint_shown = True
                        print(
                            "[warn] audio backend seems unavailable. "
                            "Install ffmpeg and/or fix torchaudio runtime for full coverage."
                        )
                    continue
                raise

            try:
                valid_samples += 1
                student_inp = self._build_inputs(
                    prompt,
                    noisy,
                    audio_path=noisy_path,
                    target_device=self.student_device,
                )

                prompt_ids = student_inp.get("input_ids")
                if prompt_ids is None:
                    raise RuntimeError("input_ids missing")
                prompt_len = int(prompt_ids.shape[1])

                candidates = self._sample_candidates(student_inp, prompt)

                # Get target answer and teacher response from dataset
                target = sample.get("target", "").strip().lower()
                teacher_response = sample.get("teacher_response", "").strip().lower()

                # Decode all candidates first
                responses = [self._decode_response(seq, prompt_len) for seq in candidates]

                # Batch score all responses at once (much faster!)
                s_lps = self._score_responses_batch(
                    self.student,
                    prompt,
                    responses,
                    noisy,
                    audio_path=noisy_path,
                    target_device=self.student_device,
                    prompt_ids=prompt_ids,
                )
                t_lps = self._score_responses_batch(
                    self.teacher,
                    prompt,
                    responses,
                    clean,
                    audio_path=clean_path,
                    target_device=self.teacher_device,
                    prompt_ids=(
                        clean_path_to_teacher_prompt_ids.get(clean_path)
                        if clean_path_to_teacher_prompt.get(clean_path) == prompt
                        else None
                    ),
                )
                t_lps = [lp.detach().to(self.student_device) for lp in t_lps]

                # Compute rewards
                choices = [c.strip().lower() for c in sample.get("choices", [])]
                rewards = [
                    self._compute_response_reward(
                        response,
                        target=target,
                        teacher_response=teacher_response,
                        choices=choices,
                        wrong_choice_reward=-1.0,
                    )
                    for response in responses
                ]

                rewards_t = torch.tensor(rewards, device=self.student_device, dtype=torch.float32)

                # Clamp rewards to reasonable range [-1, 2] to allow gradient flow
                rewards_t = torch.clamp(rewards_t, min=-1.0, max=2.0)

                s_lps_t = self._ensure_finite(torch.stack(s_lps), "student_logprobs")
                t_lps_t = self._ensure_finite(torch.stack(t_lps), "teacher_logprobs")

                # === 方式3: 多模态奖励机制改进 ===
                # 先计算蒸馏需要的forward（在policy loss之前）
                best_idx = int(torch.argmax(t_lps_t).item())

                # Use pre-generated teacher response for this clean audio
                teacher_response = clean_path_to_teacher_response.get(clean_path)
                if teacher_response is None:
                    best_seq = candidates[best_idx]
                    best_response = self._decode_response(best_seq, prompt_len)
                    teacher_response = best_response

                # Build assistant-aware inputs for distillation.
                if self._is_qwen_omni_runtime and self.processor is not None and not self.cfg.model.force_text_only_omni:
                    s_inputs = self._build_qwen_dialog_inputs(
                        prompt,
                        noisy,
                        audio_path=noisy_path,
                        assistant_text=teacher_response,
                        add_generation_prompt=False,
                        target_device=self.student_device,
                    )
                    t_inputs = self._build_qwen_dialog_inputs(
                        prompt,
                        clean,
                        audio_path=clean_path,
                        assistant_text=teacher_response,
                        add_generation_prompt=False,
                        target_device=self.teacher_device,
                    )
                    p_ids = prompt_ids
                else:
                    s_inputs = self._build_inputs(
                        prompt + teacher_response,
                        noisy,
                        audio_path=noisy_path,
                        target_device=self.student_device,
                    )
                    t_inputs = self._build_inputs(
                        prompt + teacher_response,
                        clean,
                        audio_path=clean_path,
                        target_device=self.teacher_device,
                    )
                    p_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.student_device)

                s_ids = s_inputs.get("input_ids")
                if s_ids is None:
                    s_ids = self.tokenizer(prompt + best_response, return_tensors="pt", truncation=True).input_ids.to(self.student_device)
                    s_inputs = {"input_ids": s_ids}
                if p_ids is None:
                    p_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True).input_ids.to(self.student_device)

                p_len = int(min(p_ids.shape[1], s_ids.shape[1]))
                best_mask = self._mask_response(s_ids, p_len)

                s_inputs = self._sanitize_model_inputs(s_inputs, "distill.student_inputs")
                t_inputs = self._sanitize_model_inputs(t_inputs, "distill.teacher_inputs")
                s_out = self.student(**s_inputs)
                student_logits = self._sanitize_floating_tensor(s_out.logits, "distill.student_logits")
                s_out.logits = student_logits
                t_out = self._teacher_forward_eval_mode(t_inputs)
                teacher_logits = self._sanitize_floating_tensor(
                    t_out.logits.to(self.student_device),
                    "distill.teacher_logits",
                )

                # 在policy loss计算之前，先计算多模态奖励增强
                enable_multimodal = getattr(self.cfg.sdpo, 'enable_multimodal_reward', False)
                if enable_multimodal and self.cfg.sdpo.policy_loss_weight > 0:
                    audio_weight = getattr(self.cfg.sdpo, 'reward_audio_weight', 0.5)
                    # 使用KL散度作为音频相似度的代理
                    s_logp = F.log_softmax(student_logits[:, :-1, :], dim=-1)
                    t_logp = F.log_softmax(teacher_logits[:, :-1, :], dim=-1)
                    kl_sim = F.kl_div(s_logp, t_logp, reduction="none", log_target=True).sum(-1).mean()
                    # 将KL转换为相似度 (KL越小，相似度越高)
                    audio_sim = torch.exp(-kl_sim).detach()  # 范围 [0, 1]

                    # 对当前sample的所有正确答案添加音频相似度作为额外credit
                    for idx in range(len(rewards)):
                        if rewards[idx] > 0:
                            # 新reward = 原始reward + audio_sim * audio_weight
                            rewards_t[idx] = rewards[idx] + audio_sim * audio_weight

                    # 重新clamp以防超出范围
                    rewards_t = torch.clamp(rewards_t, min=-1.0, max=2.0)

                # Only compute policy loss if policy_loss_weight > 0
                if self.cfg.sdpo.policy_loss_weight > 0:
                    is_clip = getattr(self.cfg.sdpo, 'is_clip', None)
                    policy_old_logps = t_lps_t
                    if is_clip and is_clip > 0:
                        p_loss = self._ensure_finite(
                            grpo_policy_loss(s_lps_t, rewards_t, old_logps=policy_old_logps, is_clip=is_clip),
                            "policy_loss"
                        )
                    else:
                        p_loss = self._ensure_finite(
                            grpo_policy_loss(s_lps_t, rewards_t, old_logps=policy_old_logps),
                            "policy_loss"
                        )
                    policy_list.append(p_loss)
                else:
                    # When policy_loss_weight is 0, use 0 as placeholder
                    policy_list.append(torch.tensor(0.0, device=self.student_device))

                reward_list.append(float(rewards_t.mean().item()))

                d_loss = self._ensure_finite(distill_kl_loss(student_logits, teacher_logits, best_mask), "distill_loss")
                distill_list.append(d_loss)
            except SampleSkipError as e:
                skipped_samples += 1
                valid_samples -= 1
                self._warn_skip_sample(sample, e, "unavailable multimodal path")
                continue
            except Exception as e:
                if self._is_oom_error(e) or self._is_fatal_cuda_error(e):
                    raise
                if self.cfg.data.skip_bad_audio:
                    skipped_samples += 1
                    valid_samples -= 1
                    self._warn_skip_sample(sample, e, "runtime error")
                    continue
                raise

        policy = torch.stack(policy_list).mean() if policy_list else torch.tensor(0.0, device=self.student_device)
        distill = torch.stack(distill_list).mean() if distill_list else torch.tensor(0.0, device=self.student_device)
        feedback = torch.tensor(0.0, device=self.student_device)

        total = (
            self.cfg.sdpo.policy_loss_weight * policy
            + self.cfg.sdpo.distill_loss_weight * distill
        )
        reward_mean = sum(reward_list) / max(1, len(reward_list))
        if reward_mean != reward_mean:
            reward_mean = 0.0

        return StepOutput(
            total=total,
            policy=policy,
            distill=distill,
            feedback=feedback,
            reward_mean=reward_mean,
            skipped_samples=skipped_samples,
            valid_samples=valid_samples,
        )

    def _get_eval_subset_batches(self, val_loader: DataLoader) -> List[List[Dict[str, Any]]]:
        subset_size = int(getattr(self.cfg.train, "eval_subset_size", 0) or 0)
        batch_size = max(1, int(getattr(val_loader, "batch_size", 1) or 1))
        dataset = getattr(val_loader, "dataset", None)

        if dataset is None or subset_size <= 0:
            return list(val_loader)

        dataset_len = len(dataset)
        if dataset_len <= subset_size:
            return list(val_loader)

        cache_key = (id(dataset), dataset_len, subset_size)
        if self._eval_subset_cache_key != cache_key or self._eval_subset_indices is None:
            self._eval_subset_indices = self._build_balanced_eval_subset_indices(dataset, subset_size)
            self._eval_subset_cache_key = cache_key

        subset_rows = [dataset[idx] for idx in self._eval_subset_indices]
        return [subset_rows[i:i + batch_size] for i in range(0, len(subset_rows), batch_size)]

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        was_training = self.student.training
        self.student.eval()

        noisy_correct = 0
        clean_correct = 0
        consistency_correct = 0
        consistency_total = 0
        total = 0
        skipped = 0
        empty_noisy = 0
        empty_clean = 0

        try:
            eval_batches = self._get_eval_subset_batches(val_loader)
            for batch in eval_batches:
                all_audio_paths = []
                for sample in batch:
                    noisy_path = sample.get("noisy_audio_path", "")
                    clean_path = sample.get("clean_audio_path", "")
                    if noisy_path:
                        all_audio_paths.append(noisy_path)
                    if clean_path:
                        all_audio_paths.append(clean_path)
                audio_results = {}
                if all_audio_paths:
                    results = self._load_audio_parallel(all_audio_paths)
                    audio_results = {path: results.get(path) for path in all_audio_paths}

                for sample in batch:
                    noisy_path = sample.get("noisy_audio_path", "")
                    clean_path = sample.get("clean_audio_path", "")
                    prompt = sample.get("prompt", "")

                    try:
                        noisy = audio_results.get(noisy_path)
                        clean = audio_results.get(clean_path)
                        if noisy is None or clean is None:
                            raise SampleSkipError("validation audio load failed")

                        noisy_inp = self._build_inputs(
                            prompt,
                            noisy,
                            audio_path=noisy_path,
                            target_device=self.student_device,
                        )
                        prompt_ids = noisy_inp.get("input_ids")
                        if prompt_ids is None:
                            raise SampleSkipError("validation input_ids missing")
                        prompt_len = int(prompt_ids.shape[1])

                        clean_inp = self._build_inputs(
                            prompt,
                            clean,
                            audio_path=clean_path,
                            target_device=self.student_device,
                        )
                        clean_prompt_ids = clean_inp.get("input_ids")
                        if clean_prompt_ids is None:
                            raise SampleSkipError("validation clean input_ids missing")

                        noisy_response = self._generate_student_response(noisy_inp, prompt_len)
                        clean_response = self._generate_student_response(clean_inp, int(clean_prompt_ids.shape[1]))

                        if not self._normalize_eval_text(noisy_response):
                            empty_noisy += 1
                        if not self._normalize_eval_text(clean_response):
                            empty_clean += 1

                        noisy_answer = self._canonicalize_eval_response(noisy_response, sample)
                        clean_answer = self._canonicalize_eval_response(clean_response, sample)

                        total += 1
                        if self._response_is_correct(noisy_response, sample):
                            noisy_correct += 1
                        if self._response_is_correct(clean_response, sample):
                            clean_correct += 1
                        if noisy_answer and clean_answer:
                            consistency_total += 1
                            if noisy_answer == clean_answer:
                                consistency_correct += 1
                    except SampleSkipError as e:
                        skipped += 1
                        self._warn_skip_sample(sample, e, "eval skip")
                    except Exception as e:
                        if self._is_oom_error(e):
                            self._clear_cuda_cache()
                            skipped += 1
                            self._warn_skip_sample(sample, e, "eval oom")
                            continue
                        if self._is_fatal_cuda_error(e):
                            raise
                        skipped += 1
                        self._warn_skip_sample(sample, e, "eval runtime error")
        finally:
            if was_training:
                self.student.train()

        accuracy = float(noisy_correct / total) if total > 0 else 0.0
        clean_accuracy = float(clean_correct / total) if total > 0 else 0.0
        noisy_vs_clean = float(consistency_correct / consistency_total) if consistency_total > 0 else 0.0
        valid_pair_rate = float(consistency_total / total) if total > 0 else 0.0
        return {
            "val_accuracy": accuracy,
            "val_clean_accuracy": clean_accuracy,
            "student_noisy_vs_student_clean": noisy_vs_clean,
            "val_valid_pair_rate": valid_pair_rate,
            "val_correct": int(noisy_correct),
            "val_clean_correct": int(clean_correct),
            "val_total": int(total),
            "val_consistency_correct": int(consistency_correct),
            "val_consistency_total": int(consistency_total),
            "val_skipped": int(skipped),
            "val_empty_noisy": int(empty_noisy),
            "val_empty_clean": int(empty_clean),
        }

    def save_checkpoint(self, step: int, extra: Dict[str, Any] | None = None) -> None:
        ckpt = Path(self.cfg.data.output_dir) / f"step_{step}"
        ckpt.mkdir(parents=True, exist_ok=True)
        model = self.student.module if hasattr(self.student, "module") else self.student
        model.save_pretrained(str(ckpt / "student"))
        self.tokenizer.save_pretrained(str(ckpt / "student"))
        torch.save({"step": step, "optimizer": self.optimizer.state_dict(), "extra": extra or {}}, ckpt / "state.pt")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load checkpoint and return the step to resume from.

        Args:
            checkpoint_path: Path to checkpoint directory (e.g., "./outputs/step_1000")

        Returns:
            The step number to resume from
        """
        ckpt = Path(checkpoint_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load student model - handle both PeftModel and base model
        student_path = ckpt / "student"
        model = self.student.module if hasattr(self.student, "module") else self.student

        # Check if it's a PeftModel
        if PeftModel is not None and isinstance(model, PeftModel):
            # For PeftModel, load the adapter weights
            model.load_adapter(str(student_path), adapter_name="default")
        else:
            # For base model, use from_pretrained
            model.from_pretrained(str(student_path))

        print(f"[info] Loaded student model from {student_path}")

        # Load optimizer state
        state_file = ckpt / "state.pt"
        if state_file.exists():
            state = torch.load(state_file, map_location="cpu")

            # Re-create optimizer after model load - bf16 dtype mismatch is complex
            # Create fresh optimizer to avoid dtype issues
            self.optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.cfg.train.learning_rate,
                weight_decay=self.cfg.train.weight_decay,
            )

            resume_step = state.get("step", 0)
            extra = state.get("extra", {})
            self._resume_step = int(resume_step)
            self._resume_epoch = int(extra.get("epoch", 1) or 1) if isinstance(extra, dict) else 1
            self._resume_batch_idx = int(extra.get("batch_idx", 0) or 0) if isinstance(extra, dict) else 0
            if extra:
                print(f"[info] Model loaded from step={resume_step}, extra={extra}")
            print("[info] Created new optimizer (bf16 dtype fix)")
            print(
                "[info] Resuming with loaded model weights; optimizer state was not restored, "
                f"so training will continue from step={resume_step} with a fresh optimizer"
            )
            return int(resume_step)
        else:
            print(f"[warn] No state.pt found in {ckpt}, starting from step 0")
            return 0

    @staticmethod
    def find_latest_checkpoint(output_dir: str) -> str | None:
        """Find the latest checkpoint in output directory.

        Args:
            output_dir: Path to output directory

        Returns:
            Path to latest checkpoint or None if no checkpoint found
        """
        output_path = Path(output_dir)
        if not output_path.exists():
            return None

        # Find all step_* directories
        step_dirs = []
        for item in output_path.iterdir():
            if item.is_dir() and item.name.startswith("step_"):
                try:
                    step_num = int(item.name.split("_")[1])
                    step_dirs.append((step_num, item))
                except (ValueError, IndexError):
                    continue

        if not step_dirs:
            return None

        # Return the one with highest step number
        step_dirs.sort(key=lambda x: x[0], reverse=True)
        latest = step_dirs[0][1]
        print(f"[info] Found latest checkpoint: {latest.name} (step={step_dirs[0][0]})")
        return str(latest)

    def log_metrics(self, record: Dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _iter_train_microbatches(self, batch: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        return [batch]

    def _merge_train_step_summaries(
        self,
        summaries: List[Dict[str, Any]],
        *,
        skipped_samples: int,
    ) -> StepOutput:
        device = self.student_device
        if not summaries:
            zero = torch.tensor(0.0, device=device)
            return StepOutput(
                total=zero,
                policy=zero,
                distill=zero,
                feedback=zero,
                reward_mean=0.0,
                skipped_samples=skipped_samples,
                valid_samples=0,
            )

        total_weight = float(sum(max(1, int(item["valid_samples"])) for item in summaries))
        if total_weight <= 0:
            total_weight = float(len(summaries))

        def _weighted(name: str) -> torch.Tensor:
            acc = torch.tensor(0.0, device=device)
            for item in summaries:
                weight = float(max(1, int(item["valid_samples"])))
                acc = acc + item[name].to(device) * weight
            return acc / total_weight

        reward_mean = 0.0
        for item in summaries:
            weight = float(max(1, int(item["valid_samples"])))
            reward_mean += float(item["reward_mean"]) * weight
        reward_mean /= total_weight

        return StepOutput(
            total=_weighted("total"),
            policy=_weighted("policy"),
            distill=_weighted("distill"),
            feedback=_weighted("feedback"),
            reward_mean=reward_mean,
            skipped_samples=skipped_samples,
            valid_samples=int(sum(int(item["valid_samples"]) for item in summaries)),
        )

    def _student_trainable_parameters(self) -> List[torch.nn.Parameter]:
        return [param for param in self.student.parameters() if bool(getattr(param, "requires_grad", False))]

    def _stash_param_grads(
        self,
        params: List[torch.nn.Parameter],
    ) -> List[Optional[torch.Tensor]]:
        saved_grads: List[Optional[torch.Tensor]] = []
        for param in params:
            grad = getattr(param, "grad", None)
            saved_grads.append(grad.detach().clone() if grad is not None else None)
            param.grad = None
        return saved_grads

    def _scale_current_param_grads(
        self,
        params: List[torch.nn.Parameter],
        *,
        scale: float,
    ) -> None:
        alpha = float(scale)
        for param in params:
            grad = getattr(param, "grad", None)
            if grad is None:
                continue
            grad.mul_(alpha)

    def _merge_stashed_param_grads(
        self,
        params: List[torch.nn.Parameter],
        saved_grads: List[Optional[torch.Tensor]],
    ) -> None:
        for param, saved_grad in zip(params, saved_grads):
            if saved_grad is None:
                continue
            if param.grad is None:
                param.grad = saved_grad
            else:
                param.grad.add_(saved_grad)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None, resume_step: int = 0) -> None:
        self.student.train()

        if resume_step > 0:
            print(f"[info] Resuming training from step {resume_step}")

        use_amp = torch.cuda.is_available() and self.cfg.model.mixed_precision in {"fp16", "bf16"}
        scaler = torch.cuda.amp.GradScaler() if use_amp and self.cfg.model.mixed_precision == "fp16" else None

        step = 0
        last = time.time()
        consecutive_oom = 0
        trainable_params = self._student_trainable_parameters()

        for epoch in range(1, self.cfg.train.num_epochs + 1):
            last_batch_idx = 0
            if tqdm is not None:
                total_batches = len(train_loader) * self.cfg.train.num_epochs
                train_loader_iter = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch}/{self.cfg.train.num_epochs}",
                    initial=epoch - 1,
                    total=total_batches,
                    unit="batch",
                )
                inner_enumerate = enumerate(train_loader_iter, start=1)
            else:
                train_loader_iter = None
                inner_enumerate = enumerate(train_loader, start=1)

            for batch_idx, batch in inner_enumerate:
                last_batch_idx = batch_idx
                # Skip batches if resuming from checkpoint
                if resume_step > 0 and step < resume_step:
                    step += 1
                    if step % 100 == 0:
                        print(f"[info] Skipping batch {step}/{resume_step}")
                    continue

                step += 1
                if self.cfg.oom.enabled and self.cfg.oom.clear_cache_interval > 0 and step % self.cfg.oom.clear_cache_interval == 0:
                    self._clear_cuda_cache()

                try:
                    micro_batches = self._iter_train_microbatches(batch)
                    total_batch_slots = max(1, sum(len(micro_batch) for micro_batch in micro_batches))
                    use_exact_valid_weighting = len(micro_batches) > 1 and bool(trainable_params)
                    stashed_param_grads: List[Optional[torch.Tensor]] = (
                        self._stash_param_grads(trainable_params) if use_exact_valid_weighting else []
                    )
                    batch_valid_grad_samples = 0
                    batch_skipped_samples = 0
                    out_summaries: List[Dict[str, Any]] = []
                    had_valid_microbatch = False
                    had_loss_skip = False
                    discard_stashed_grads = False

                    for micro_batch in micro_batches:
                        if use_amp and self.cfg.model.mixed_precision == "fp16":
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                micro_out = self.compute_step(micro_batch)
                                if micro_out.valid_samples == 0:
                                    batch_skipped_samples += int(micro_out.skipped_samples)
                                    continue
                                batch_skipped_samples += int(micro_out.skipped_samples)
                                had_valid_microbatch = True
                                loss = micro_out.total / self.cfg.train.gradient_accumulation_steps
                                if self._should_skip_loss(loss, epoch, batch_idx, step):
                                    had_loss_skip = True
                                    if use_exact_valid_weighting and self._last_loss_skip_event == "nonfinite_loss_skip":
                                        discard_stashed_grads = True
                                        batch_valid_grad_samples = 0
                                    if train_loader_iter is not None:
                                        train_loader_iter.set_postfix({"event": self._last_loss_skip_event or "loss_skip"})
                                    continue
                            micro_weight = float(len(micro_batch)) / float(total_batch_slots)
                            assert scaler is not None
                            scaler.scale(micro_out.total * micro_weight / self.cfg.train.gradient_accumulation_steps).backward()
                            if use_exact_valid_weighting:
                                batch_valid_grad_samples += int(micro_out.valid_samples)
                            self._after_backward(step, micro_batch, micro_out)
                        elif use_amp and self.cfg.model.mixed_precision == "bf16":
                            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                                micro_out = self.compute_step(micro_batch)
                                if micro_out.valid_samples == 0:
                                    batch_skipped_samples += int(micro_out.skipped_samples)
                                    continue
                                batch_skipped_samples += int(micro_out.skipped_samples)
                                had_valid_microbatch = True
                                loss = micro_out.total / self.cfg.train.gradient_accumulation_steps
                                if self._should_skip_loss(loss, epoch, batch_idx, step):
                                    had_loss_skip = True
                                    if use_exact_valid_weighting and self._last_loss_skip_event == "nonfinite_loss_skip":
                                        discard_stashed_grads = True
                                        batch_valid_grad_samples = 0
                                    if train_loader_iter is not None:
                                        train_loader_iter.set_postfix({"event": self._last_loss_skip_event or "loss_skip"})
                                    continue
                            micro_weight = float(len(micro_batch)) / float(total_batch_slots)
                            (micro_out.total * micro_weight / self.cfg.train.gradient_accumulation_steps).backward()
                            if use_exact_valid_weighting:
                                batch_valid_grad_samples += int(micro_out.valid_samples)
                            self._after_backward(step, micro_batch, micro_out)
                        else:
                            micro_out = self.compute_step(micro_batch)
                            if micro_out.valid_samples == 0:
                                batch_skipped_samples += int(micro_out.skipped_samples)
                                continue
                            batch_skipped_samples += int(micro_out.skipped_samples)
                            had_valid_microbatch = True
                            loss = micro_out.total / self.cfg.train.gradient_accumulation_steps
                            if self._should_skip_loss(loss, epoch, batch_idx, step):
                                had_loss_skip = True
                                if use_exact_valid_weighting and self._last_loss_skip_event == "nonfinite_loss_skip":
                                    discard_stashed_grads = True
                                    batch_valid_grad_samples = 0
                                if train_loader_iter is not None:
                                    train_loader_iter.set_postfix({"event": self._last_loss_skip_event or "loss_skip"})
                                continue
                            micro_weight = float(len(micro_batch)) / float(total_batch_slots)
                            (micro_out.total * micro_weight / self.cfg.train.gradient_accumulation_steps).backward()
                            if use_exact_valid_weighting:
                                batch_valid_grad_samples += int(micro_out.valid_samples)
                            self._after_backward(step, micro_batch, micro_out)

                        out_summaries.append(
                            {
                                "total": micro_out.total.detach(),
                                "policy": micro_out.policy.detach(),
                                "distill": micro_out.distill.detach(),
                                "feedback": micro_out.feedback.detach(),
                                "reward_mean": float(micro_out.reward_mean),
                                "valid_samples": int(micro_out.valid_samples),
                            }
                        )

                    if not out_summaries:
                        if use_exact_valid_weighting and not discard_stashed_grads:
                            self._merge_stashed_param_grads(trainable_params, stashed_param_grads)
                        if had_valid_microbatch and had_loss_skip:
                            continue
                        rec = {
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "step": step,
                            "event": "all_samples_skipped",
                            "skipped_samples": int(batch_skipped_samples),
                        }
                        self.log_metrics(rec)
                        self._log_swanlab(
                            {
                                "train/all_samples_skipped": 1,
                                "train/skipped_samples": int(batch_skipped_samples),
                                "step": step,
                            }
                        )
                        continue

                    if use_exact_valid_weighting and batch_valid_grad_samples > 0 and batch_valid_grad_samples != total_batch_slots:
                        self._scale_current_param_grads(
                            trainable_params,
                            scale=float(total_batch_slots) / float(batch_valid_grad_samples),
                        )
                    if use_exact_valid_weighting and not discard_stashed_grads:
                        self._merge_stashed_param_grads(trainable_params, stashed_param_grads)

                    out = self._merge_train_step_summaries(
                        out_summaries,
                        skipped_samples=batch_skipped_samples,
                    )
                    consecutive_oom = 0
                except RuntimeError as e:
                    if self._is_fatal_cuda_error(e):
                        print(
                            "[warn] fatal cuda during fit "
                            f"step={step} epoch={epoch} batch_idx={batch_idx} "
                            f"had_valid_microbatch={had_valid_microbatch} "
                            f"batch_valid_grad_samples={batch_valid_grad_samples} "
                            f"batch_skipped_samples={batch_skipped_samples} "
                            f"last_zero_detail={self._last_zero_loss_detail}"
                        )
                        raise
                    if not (self.cfg.oom.enabled and self._is_oom_error(e)):
                        raise

                    consecutive_oom += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    self._clear_cuda_cache()

                    rec = {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "step": step,
                        "event": "oom_skip",
                        "consecutive_oom": consecutive_oom,
                    }
                    self.log_metrics(rec)
                    self._log_swanlab({"train/oom_skip": 1, "train/consecutive_oom": consecutive_oom, "step": step})
                    print(f"[warn] OOM at step={step}, skip current batch (consecutive={consecutive_oom})")

                    if consecutive_oom >= self.cfg.oom.max_consecutive_oom:
                        raise RuntimeError(
                            f"Exceeded max consecutive OOM: {self.cfg.oom.max_consecutive_oom}. "
                            "Please reduce batch_size/max_new_tokens/group_size."
                        ) from e

                    if self.cfg.oom.skip_on_oom:
                        continue
                    raise

                if step % self.cfg.train.gradient_accumulation_steps == 0:
                    if scaler is not None:
                        scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.train.max_grad_norm)
                        scaler.step(self.optimizer)
                        scaler.update()
                        self._after_optimizer_step(step)
                    else:
                        torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.train.max_grad_norm)
                        self.optimizer.step()
                        self._after_optimizer_step(step)
                    self.optimizer.zero_grad(set_to_none=True)

                # Update progress bar with current loss
                    if train_loader_iter is not None:
                        train_loader_iter.set_postfix({
                            "loss": f"{float(out.total.item()):.4f}",
                            "policy": f"{float(out.policy.item()):.4f}",
                            "distill": f"{float(out.distill.item()):.4f}",
                        })

                if val_loader is not None and step % self.cfg.train.eval_interval == 0:
                    v = self.evaluate(val_loader)
                    self._last_val_metrics = {
                        "val_accuracy": float(v["val_accuracy"]),
                        "val_clean_accuracy": float(v["val_clean_accuracy"]),
                        "student_noisy_vs_student_clean": float(v["student_noisy_vs_student_clean"]),
                        "val_valid_pair_rate": float(v.get("val_valid_pair_rate", 0.0)),
                        "val_empty_noisy": float(v.get("val_empty_noisy", 0.0)),
                        "val_empty_clean": float(v.get("val_empty_clean", 0.0)),
                    }
                    self.log_metrics({"step": step, **v})
                    self._log_swanlab(
                        {
                            "val/accuracy": v["val_accuracy"],
                            "val/clean_accuracy": v["val_clean_accuracy"],
                            "val/student_noisy_vs_student_clean": v["student_noisy_vs_student_clean"],
                            "val/valid_pair_rate": v.get("val_valid_pair_rate", 0.0),
                            "val/empty_noisy": v.get("val_empty_noisy", 0),
                            "val/empty_clean": v.get("val_empty_clean", 0),
                            "step": step,
                        }
                    )
                    print(
                        f"[eval] step={step} val_accuracy={v['val_accuracy']:.4f} "
                        f"clean_accuracy={v['val_clean_accuracy']:.4f} "
                        f"student_noisy_vs_student_clean={v['student_noisy_vs_student_clean']:.4f} "
                        f"(noisy={v['val_correct']}/{v['val_total']}, "
                        f"clean={v['val_clean_correct']}/{v['val_total']}, "
                        f"consistency={v['val_consistency_correct']}/{v['val_consistency_total']}, "
                        f"valid_pair_rate={v.get('val_valid_pair_rate', 0.0):.4f}, "
                        f"skipped={v['val_skipped']}, "
                        f"empty_noisy={v.get('val_empty_noisy', 0)}, "
                        f"empty_clean={v.get('val_empty_clean', 0)})"
                    )

                if step % self.cfg.train.log_interval == 0:
                    now = time.time()
                    rec = {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "step": step,
                        "loss_total": float(out.total.item()),
                        "loss_policy": float(out.policy.item()),
                        "loss_distill": float(out.distill.item()),
                        "reward_mean": out.reward_mean,
                        "skipped_samples": int(out.skipped_samples),
                        "dt": float(now - last),
                    }
                    if self._last_val_metrics:
                        rec.update(
                            {
                                "val_accuracy": float(self._last_val_metrics["val_accuracy"]),
                                "val_clean_accuracy": float(self._last_val_metrics["val_clean_accuracy"]),
                                "student_noisy_vs_student_clean": float(
                                    self._last_val_metrics["student_noisy_vs_student_clean"]
                                ),
                                "val_valid_pair_rate": float(
                                    self._last_val_metrics.get("val_valid_pair_rate", 0.0)
                                ),
                                "val_empty_noisy": float(self._last_val_metrics.get("val_empty_noisy", 0.0)),
                                "val_empty_clean": float(self._last_val_metrics.get("val_empty_clean", 0.0)),
                            }
                        )
                    last = now
                    self.log_metrics(rec)
                    train_payload = {
                        "train/loss_total": rec["loss_total"],
                        "train/loss_policy": rec["loss_policy"],
                        "train/loss_distill": rec["loss_distill"],
                        "train/reward_mean": rec["reward_mean"],
                        "train/skipped_samples": rec["skipped_samples"],
                        "train/epoch": epoch,
                        "step": step,
                    }
                    if self._last_val_metrics:
                        train_payload.update(
                            {
                                "train/val_accuracy": rec["val_accuracy"],
                                "train/val_clean_accuracy": rec["val_clean_accuracy"],
                                "train/student_noisy_vs_student_clean": rec["student_noisy_vs_student_clean"],
                                "train/val_valid_pair_rate": rec["val_valid_pair_rate"],
                                "train/val_empty_noisy": rec["val_empty_noisy"],
                                "train/val_empty_clean": rec["val_empty_clean"],
                            }
                        )
                    self._log_swanlab(train_payload)
                    train_msg = (
                        f"[train] e={epoch} s={step} total={rec['loss_total']:.4f} "
                        f"policy={rec['loss_policy']:.4f} distill={rec['loss_distill']:.4f} "
                        f"reward={rec['reward_mean']:.4f}"
                    )
                    if self._last_val_metrics:
                        train_msg += (
                            f" val_accuracy={rec['val_accuracy']:.4f} "
                            f"clean_accuracy={rec['val_clean_accuracy']:.4f} "
                            f"student_noisy_vs_student_clean={rec['student_noisy_vs_student_clean']:.4f} "
                            f"valid_pair_rate={rec['val_valid_pair_rate']:.4f} "
                            f"empty_noisy={int(rec['val_empty_noisy'])} "
                            f"empty_clean={int(rec['val_empty_clean'])}"
                        )
                    print(train_msg)

                if step % self.cfg.train.save_interval == 0:
                    self.save_checkpoint(step, extra={"epoch": epoch, "batch_idx": batch_idx})
                    print(f"[save] step={step}")

            self.save_checkpoint(step, extra={"epoch": epoch, "batch_idx": last_batch_idx})
            print(f"[save] epoch={epoch}")

    def close(self) -> None:
        if self.swanlab is None:
            return
        try:
            self.swanlab.finish()
        except Exception:
            pass
