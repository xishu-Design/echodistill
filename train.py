from __future__ import annotations

import logging
import os
import random
import sys
import types
import warnings
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Suppress warnings BEFORE importing any modules
# This is important because transformers emits warnings during import
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

warnings.filterwarnings("ignore")
# Suppress specific warnings that are not actionable
warnings.filterwarnings("ignore", message=".*use_cache=True.*gradient checkpointing.*")
logging.basicConfig(
    level=logging.ERROR,
    format="%(levelname)s: %(message)s",
    handlers=[logging.NullHandler()]
)

# Suppress transformers logger completely
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)
logging.getLogger("torch.utils.checkpoint").setLevel(logging.ERROR)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sdpoa.config import build_arg_parser, load_config
from sdpoa.data.dataset import AudioPairDataset, audio_pair_collate
from sdpoa.data.schema import parse_path_maps
from sdpoa.models.omni_adapter import _resolve_safe_cuda_device, _sanitize_visible_gpu_ids, build_omni_bundle
from sdpoa.training.trainer import DistillationTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _iter_lora_modules(model: Any):
    for module in model.modules():
        if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
            yield module


def _sanitize_qwen_lora_tensor(tensor: Any, name: str) -> Any:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return tensor
    if torch.isfinite(tensor).all():
        return tensor
    print(f"[warn] qwen lora sanitized non-finite tensor: {name} shape={tuple(tensor.shape)}")
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _rebuild_lora_linear_modules(model: Any) -> int:
    rebuilt = 0
    for module in _iter_lora_modules(model):
        for attr in ("lora_A", "lora_B"):
            container = getattr(module, attr, None)
            if not hasattr(container, "items"):
                continue
            for key, layer in list(container.items()):
                if not hasattr(layer, "weight"):
                    continue
                device = layer.weight.device
                dtype = layer.weight.dtype
                bias_flag = getattr(layer, "bias", None) is not None
                fresh = torch.nn.Linear(
                    int(layer.in_features),
                    int(layer.out_features),
                    bias=bias_flag,
                    device=device,
                    dtype=dtype,
                )
                with torch.no_grad():
                    fresh.weight.copy_(layer.weight.detach())
                    if bias_flag and fresh.bias is not None and layer.bias is not None:
                        fresh.bias.copy_(layer.bias.detach())
                container[key] = fresh
                rebuilt += 1
    if rebuilt > 0:
        print(f"[info] rebuilt qwen LoRA linear modules: {rebuilt}")
    return rebuilt


def _rehydrate_lora_linear_params(model: Any) -> int:
    changed = 0
    for module in _iter_lora_modules(model):
        for attr in ("lora_A", "lora_B"):
            container = getattr(module, attr, None)
            if not hasattr(container, "items"):
                continue
            for _, layer in list(container.items()):
                if not hasattr(layer, "weight"):
                    continue
                layer.weight = torch.nn.Parameter(layer.weight.detach().clone(), requires_grad=True)
                if getattr(layer, "bias", None) is not None:
                    layer.bias = torch.nn.Parameter(layer.bias.detach().clone(), requires_grad=True)
                changed += 1
    if changed > 0:
        print(f"[info] rehydrated qwen LoRA params: {changed}")
    return changed


def _patch_peft_lora_modules(model: Any) -> int:
    patched = 0
    for module in _iter_lora_modules(model):
        if getattr(module, "_sdpoa_qwen_lora_forward_patched", False):
            continue
        if not hasattr(module, "base_layer"):
            continue
        module_mod = str(type(module).__module__)
        if "peft.tuners.lora" not in module_mod:
            continue
        module_cls_name = str(type(module).__name__).lower()
        if "linear" not in module_cls_name:
            continue

        original_forward = getattr(module, "forward", None)

        def _patched_forward(self, x, *args, **kwargs):
            result = self.base_layer(x, *args, **kwargs)
            result = _sanitize_qwen_lora_tensor(result, "base_layer_result")
            if bool(getattr(self, "disable_adapters", False)) or bool(getattr(self, "_disable_adapters", False)):
                return result
            if bool(getattr(self, "merged", False)):
                return result

            active = getattr(self, "active_adapters", None)
            if isinstance(active, str):
                active = [active]
            elif active is None:
                active = []
                single = getattr(self, "active_adapter", None)
                if isinstance(single, str):
                    active = [single]
            if not active:
                try:
                    active = [key for key in self.lora_A.keys() if key in self.lora_B]
                except Exception:
                    active = []
            if not active:
                return result

            target_dtype = result.dtype if torch.is_tensor(result) else x.dtype
            target_device = result.device if torch.is_tensor(result) else getattr(x, "device", None)
            base_x = x.to(dtype=target_dtype) if getattr(x, "dtype", None) != target_dtype else x
            if target_device is not None and getattr(base_x, "device", None) != target_device:
                base_x = base_x.to(device=target_device)
            for adapter in active:
                if adapter not in self.lora_A or adapter not in self.lora_B:
                    continue
                lora_a = self.lora_A[adapter]
                lora_b = self.lora_B[adapter]
                if not hasattr(lora_a, "weight") or not hasattr(lora_b, "weight"):
                    continue
                dropout = None
                try:
                    dropout = self.lora_dropout[adapter]
                except Exception:
                    dropout = None
                dropped = dropout(base_x) if dropout is not None else base_x
                weight_a = lora_a.weight
                weight_b = lora_b.weight
                bias_a = getattr(lora_a, "bias", None)
                bias_b = getattr(lora_b, "bias", None)
                if target_device is not None and (
                    getattr(weight_a, "device", None) != target_device or getattr(weight_a, "dtype", None) != target_dtype
                ):
                    weight_a = weight_a.to(device=target_device, dtype=target_dtype)
                elif getattr(weight_a, "dtype", None) != target_dtype:
                    weight_a = weight_a.to(dtype=target_dtype)
                if target_device is not None and (
                    getattr(weight_b, "device", None) != target_device or getattr(weight_b, "dtype", None) != target_dtype
                ):
                    weight_b = weight_b.to(device=target_device, dtype=target_dtype)
                elif getattr(weight_b, "dtype", None) != target_dtype:
                    weight_b = weight_b.to(dtype=target_dtype)
                if bias_a is not None and target_device is not None and (
                    getattr(bias_a, "device", None) != target_device or getattr(bias_a, "dtype", None) != target_dtype
                ):
                    bias_a = bias_a.to(device=target_device, dtype=target_dtype)
                elif bias_a is not None and getattr(bias_a, "dtype", None) != target_dtype:
                    bias_a = bias_a.to(dtype=target_dtype)
                if bias_b is not None and target_device is not None and (
                    getattr(bias_b, "device", None) != target_device or getattr(bias_b, "dtype", None) != target_dtype
                ):
                    bias_b = bias_b.to(device=target_device, dtype=target_dtype)
                elif bias_b is not None and getattr(bias_b, "dtype", None) != target_dtype:
                    bias_b = bias_b.to(dtype=target_dtype)
                hidden = F.linear(dropped, weight_a, bias_a)
                hidden = _sanitize_qwen_lora_tensor(hidden, "lora_hidden")
                update = F.linear(hidden, weight_b, bias_b)
                update = _sanitize_qwen_lora_tensor(update, "lora_update")
                scaling = 1.0
                try:
                    scaling = float(self.scaling.get(adapter, 1.0))
                except Exception:
                    scaling = 1.0
                result = result + update.to(dtype=result.dtype) * scaling
                result = _sanitize_qwen_lora_tensor(result, "lora_result")
            return result

        _patched_forward.__wrapped__ = original_forward
        module.forward = types.MethodType(_patched_forward, module)
        try:
            if hasattr(module, "_compiled_call_impl"):
                module._compiled_call_impl = None
        except Exception:
            pass
        module._sdpoa_qwen_lora_forward_patched = True
        patched += 1
    if patched > 0:
        print(f"[info] patched qwen PEFT LoRA forwards: {patched}")
    return patched


def _activate_lora_for_training(model: Any, adapter_name: str = "default") -> None:
    try:
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
    except Exception:
        pass
    try:
        if hasattr(model, "enable_adapter_layers"):
            model.enable_adapter_layers()
        elif hasattr(model, "base_model") and hasattr(model.base_model, "enable_adapter_layers"):
            model.base_model.enable_adapter_layers()
    except Exception:
        pass
    try:
        model.train()
    except Exception:
        pass

    for module in _iter_lora_modules(model):
        try:
            if hasattr(module, "enable_adapters"):
                module.enable_adapters(True)
        except Exception:
            pass
        try:
            if hasattr(module, "set_adapter"):
                module.set_adapter(adapter_name)
        except Exception:
            pass
        if hasattr(module, "disable_adapters"):
            try:
                module.disable_adapters = False
            except Exception:
                pass
        if hasattr(module, "_disable_adapters"):
            try:
                module._disable_adapters = False
            except Exception:
                pass
        if bool(getattr(module, "merged", False)) and hasattr(module, "unmerge"):
            try:
                module.unmerge()
            except Exception:
                pass
        for attr in ("lora_A", "lora_B"):
            container = getattr(module, attr, None)
            if not hasattr(container, "items"):
                continue
            for _, layer in list(container.items()):
                if hasattr(layer, "weight"):
                    try:
                        layer.weight.requires_grad_(True)
                    except Exception:
                        pass
                if getattr(layer, "bias", None) is not None:
                    try:
                        layer.bias.requires_grad_(True)
                    except Exception:
                        pass


def _summarize_lora_trainability(model: Any, role: str) -> None:
    total = 0
    trainable = 0
    lora_a_total = 0
    lora_a_trainable = 0
    lora_b_total = 0
    lora_b_trainable = 0
    for name, param in model.named_parameters():
        low = name.lower()
        if "lora_" not in low:
            continue
        total += 1
        if bool(getattr(param, "requires_grad", False)):
            trainable += 1
        if "lora_a" in low:
            lora_a_total += 1
            if bool(getattr(param, "requires_grad", False)):
                lora_a_trainable += 1
        if "lora_b" in low:
            lora_b_total += 1
            if bool(getattr(param, "requires_grad", False)):
                lora_b_trainable += 1
    if total > 0:
        print(
            "[diag] lora_trainability "
            f"role={role} "
            f"trainable={trainable}/{total} "
            f"lora_A={lora_a_trainable}/{lora_a_total} "
            f"lora_B={lora_b_trainable}/{lora_b_total}"
        )


def _refresh_qwen_lora_runtime(model: Any, role: str) -> None:
    if model is None:
        return
    rebuilt = _rebuild_lora_linear_modules(model)
    changed = _rehydrate_lora_linear_params(model)
    patched = _patch_peft_lora_modules(model)
    _activate_lora_for_training(model, adapter_name="default")
    _summarize_lora_trainability(model, role=role)
    if rebuilt == 0 and changed == 0 and patched == 0:
        print(f"[info] qwen LoRA runtime already active role={role}")


def _grad_summary(param: Optional[torch.nn.Parameter]) -> tuple[bool, bool, float]:
    if param is None:
        return False, False, 0.0
    grad = getattr(param, "grad", None)
    if grad is None:
        return False, False, 0.0
    grad_detached = grad.detach()
    try:
        nonzero = bool(torch.count_nonzero(grad_detached).item())
    except Exception:
        nonzero = False
    try:
        norm = float(grad_detached.float().norm().item())
    except Exception:
        norm = 0.0
    return True, nonzero, norm


def _param_delta_summary(
    param: Optional[torch.nn.Parameter],
    previous: Optional[torch.Tensor],
) -> tuple[Optional[torch.Tensor], bool, float, float]:
    if param is None:
        return previous, False, 0.0, 0.0
    current = param.detach().float().cpu().clone()
    if previous is None or previous.shape != current.shape:
        return current, False, 0.0, float(current.norm().item())
    delta = (current - previous).abs()
    try:
        nonzero = bool(torch.count_nonzero(delta).item())
    except Exception:
        nonzero = False
    try:
        delta_max = float(delta.max().item())
    except Exception:
        delta_max = 0.0
    try:
        current_norm = float(current.norm().item())
    except Exception:
        current_norm = 0.0
    return current, nonzero, delta_max, current_norm


def _sample_name_from_batch(batch: list[dict[str, Any]]) -> str:
    if not batch:
        return "unknown"
    noisy_path = str(batch[0].get("noisy_audio_path", "") or "")
    if noisy_path:
        return Path(noisy_path).name
    prompt = str(batch[0].get("prompt", "") or "").strip()
    return prompt[:60] if prompt else "unknown"


def _init_qwen_lora_probe_state(trainer: Any) -> None:
    trainer._qwen_lora_probe_a_name = ""
    trainer._qwen_lora_probe_b_name = ""
    trainer._qwen_lora_probe_a_param = None
    trainer._qwen_lora_probe_b_param = None
    trainer._qwen_lora_light_step_index = 0
    trainer._qwen_lora_optimizer_step_index = 0
    trainer._last_qwen_lora_a_snapshot = None
    trainer._last_qwen_lora_b_snapshot = None

    model = getattr(trainer, "student", None)
    if model is None:
        return

    for name, param in model.named_parameters():
        low = name.lower()
        if trainer._qwen_lora_probe_a_param is None and "lora_a" in low:
            trainer._qwen_lora_probe_a_name = name
            trainer._qwen_lora_probe_a_param = param
        if trainer._qwen_lora_probe_b_param is None and "lora_b" in low:
            trainer._qwen_lora_probe_b_name = name
            trainer._qwen_lora_probe_b_param = param
        if trainer._qwen_lora_probe_a_param is not None and trainer._qwen_lora_probe_b_param is not None:
            break


def _patch_trainer_lora_logging() -> None:
    if getattr(DistillationTrainer, "_sdpoa_qwen_lora_logging_patched", False):
        return
    original_after_backward = DistillationTrainer._after_backward
    original_after_optimizer_step = DistillationTrainer._after_optimizer_step
    log_every = 10

    def _patched_after_backward(self, step: int, batch: list[dict[str, Any]], out: Any) -> None:
        original_after_backward(self, step, batch, out)
        if not bool(getattr(self.student, "training", False)):
            return
        if not hasattr(self, "_qwen_lora_probe_b_param"):
            _init_qwen_lora_probe_state(self)
        self._qwen_lora_light_step_index += 1
        if step % log_every != 0:
            return
        a_grad = _grad_summary(getattr(self, "_qwen_lora_probe_a_param", None))
        b_grad = _grad_summary(getattr(self, "_qwen_lora_probe_b_param", None))
        sample_name = _sample_name_from_batch(batch)
        print(
            "[diag] sample_lora "
            f"sample={sample_name} "
            f"train_step={step} "
            f"microstep={self._qwen_lora_light_step_index} "
            f"loss_total={float(out.total.item()):.4f} "
            f"loss_policy={float(out.policy.item()):.4f} "
            f"loss_distill={float(out.distill.item()):.4f} "
            f"lora_A_grad={a_grad[0]} lora_A_nonzero={a_grad[1]} lora_A_norm={a_grad[2]:.6f} "
            f"lora_B_grad={b_grad[0]} lora_B_nonzero={b_grad[1]} lora_B_norm={b_grad[2]:.6f}"
        )

    def _patched_after_optimizer_step(self, step: int) -> None:
        original_after_optimizer_step(self, step)
        if not hasattr(self, "_qwen_lora_probe_b_param"):
            _init_qwen_lora_probe_state(self)
        self._qwen_lora_optimizer_step_index += 1
        self._last_qwen_lora_a_snapshot, a_changed, a_delta_max, a_norm = _param_delta_summary(
            getattr(self, "_qwen_lora_probe_a_param", None),
            getattr(self, "_last_qwen_lora_a_snapshot", None),
        )
        self._last_qwen_lora_b_snapshot, b_changed, b_delta_max, b_norm = _param_delta_summary(
            getattr(self, "_qwen_lora_probe_b_param", None),
            getattr(self, "_last_qwen_lora_b_snapshot", None),
        )
        if step % log_every != 0:
            return
        print(
            "[diag] lora_update "
            f"train_step={step} "
            f"optimizer_step={self._qwen_lora_optimizer_step_index} "
            f"lora_A_changed={a_changed} lora_A_delta_max={a_delta_max:.6e} lora_A_norm={a_norm:.6f} "
            f"lora_B_changed={b_changed} lora_B_delta_max={b_delta_max:.6e} lora_B_norm={b_norm:.6f}"
        )

    DistillationTrainer._after_backward = _patched_after_backward
    DistillationTrainer._after_optimizer_step = _patched_after_optimizer_step
    DistillationTrainer._sdpoa_qwen_lora_logging_patched = True

def _patch_trainer_checkpoint_for_lora() -> None:
    if getattr(DistillationTrainer, "_sdpoa_qwen_lora_checkpoint_patched", False):
        return
    original_load_checkpoint = DistillationTrainer.load_checkpoint

    def _patched_load_checkpoint(self, checkpoint_path: str) -> int:
        resume_step = original_load_checkpoint(self, checkpoint_path)
        _refresh_qwen_lora_runtime(self.student, role="student_after_checkpoint_resume")
        _init_qwen_lora_probe_state(self)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.cfg.train.learning_rate,
            weight_decay=self.cfg.train.weight_decay,
        )
        print("[info] rebuilt optimizer after qwen checkpoint LoRA refresh")
        return resume_step

    DistillationTrainer.load_checkpoint = _patched_load_checkpoint
    DistillationTrainer._sdpoa_qwen_lora_checkpoint_patched = True


def main() -> None:
    _patch_trainer_checkpoint_for_lora()
    _patch_trainer_lora_logging()

    parser = build_arg_parser()
    args = parser.parse_args()

    cli = vars(args).copy()
    config_path = cli.pop("config", "")
    cfg = load_config(config_path, cli)

    if not cfg.data.train_data:
        raise ValueError("data.train_data is required")

    set_seed(cfg.train.seed)

    output_dir = Path(cfg.data.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_student_gpu_ids = getattr(cfg.model, "student_gpu_ids", [])
    student_gpu_ids = _sanitize_visible_gpu_ids(raw_student_gpu_ids, role="student")
    print(
        "[info] cuda visibility "
        f"config_path={config_path} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
        f"torch_cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"raw_student_gpu_ids={raw_student_gpu_ids} "
        f"student_gpu_ids={student_gpu_ids} "
        f"teacher_gpu_id={getattr(cfg.model, 'teacher_gpu_id', -1)} "
        f"multi_gpu_dispatch={getattr(cfg.model, 'multi_gpu_dispatch', 'balanced')}"
    )
    if torch.cuda.is_available():
        if student_gpu_ids:
            device = _resolve_safe_cuda_device(student_gpu_ids[0], torch.device("cuda:0"), role="student")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[info] device={device}")

    bundle = build_omni_bundle(
        student_model_name=cfg.model.student_model,
        teacher_model_name=cfg.model.teacher_model,
        use_separate_teacher=cfg.model.use_separate_teacher,
        mixed_precision=cfg.model.mixed_precision,
        use_flash_attention=cfg.model.use_flash_attention,
        device=device,
        lora_config=cfg.lora,
        student_gpu_ids=student_gpu_ids,
        teacher_gpu_id=getattr(cfg.model, "teacher_gpu_id", -1),
        multi_gpu_dispatch=getattr(cfg.model, "multi_gpu_dispatch", "balanced"),
        gpu_memory_reserve_gb=getattr(cfg.model, "gpu_memory_reserve_gb", 4.0),
    )
    _refresh_qwen_lora_runtime(bundle.student_model, role="student_after_train_entry")

    path_maps = parse_path_maps(cfg.data.path_maps)
    train_ds = AudioPairDataset(cfg.data.train_data, path_maps)
    val_ds = AudioPairDataset(cfg.data.val_data, path_maps) if cfg.data.val_data else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=audio_pair_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=max(0, min(cfg.data.num_workers, 2)),
            collate_fn=audio_pair_collate,
            pin_memory=torch.cuda.is_available(),
        )

    trainer = DistillationTrainer(cfg=cfg, bundle=bundle, device=device)
    _init_qwen_lora_probe_state(trainer)
    resume_step = 0
    resume_from = str(getattr(cfg.data, "resume_from", "") or "").strip()
    if resume_from:
        if resume_from.lower() in {"latest", "auto"}:
            latest_ckpt = trainer.find_latest_checkpoint(cfg.data.output_dir)
            if latest_ckpt is None:
                print(f"[warn] data.resume_from={resume_from} but no checkpoint found in {cfg.data.output_dir}")
                resume_from = ""
            else:
                resume_from = latest_ckpt
        if resume_from:
            print(f"[info] loading checkpoint for resume: {resume_from}")
            resume_step = int(trainer.load_checkpoint(resume_from) or 0)

    try:
        trainer.fit(train_loader=train_loader, val_loader=val_loader, resume_step=resume_step)
    finally:
        trainer.close()
    print("[done] training finished")


if __name__ == "__main__":
    main()
