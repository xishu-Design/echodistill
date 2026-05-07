# EchoDistill Release

This repository contains a cleaned release of the EchoDistill training code and JSONL dataset
metadata for noisy-to-clean audio self-distillation.

## What is included

```text
.
├── configs/default.yaml
├── data/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── dataset_stats.json
│   └── sample_instance.json
├── sdpoa/
│   ├── config.py
│   ├── data/
│   │   ├── dataset.py
│   │   └── schema.py
│   ├── models/
│   │   └── omni_adapter.py
│   ├── training/
│   │   ├── losses.py
│   │   └── trainer.py
│   └── utils/
│       └── audio.py
├── train.py
├── requirements.txt
└── DATASET_CARD.md
```

## Dataset format

Each JSONL row contains the static fields needed for training:

```json
{
  "id": 19452,
  "prompt": "What is producing the sound in the audio?...",
  "noisy_audio_path": "noise/water/snr_30/audio_noise/example.wav",
  "clean_audio_path": "audio/example.wav",
  "choices": ["Airplane", "Motorcycle", "Train", "Sports car"],
  "target": "Airplane",
  "noise_type": "water",
  "snr": 30
}
```

The dataset intentionally does **not** store sampled candidate responses, static reward values,
or static teacher responses. Candidate responses are sampled online; rewards are computed online
from the sampled response, target answer, and choices. The clean-audio teacher response used for
distillation is generated during training.

## Audio paths

The JSONL files use relative audio paths. The audio files themselves are not included in this
release. Use `data.path_maps` in `configs/default.yaml` or pass `--data.path_map` to map these
relative paths to your local audio root.

Example:

```bash
python train.py   --config configs/default.yaml   --data.path_map "audio=/path/to/mma/audio"   --data.path_map "noise=/path/to/mma/noise"
```

## Training

Install dependencies, then run:

```bash
pip install -r requirements.txt
python train.py --config configs/default.yaml
```

## Hardware

The experiments in this work were conducted on 2 GPUs with 80GB memory each.
Depending on the available hardware, users may adjust the batch size, gradient accumulation steps, group size, maximum generation length, precision, LoRA/QLoRA settings, and GPU dispatch options in `configs/default.yaml`.

Useful overrides:

```bash
python train.py   --config configs/default.yaml   --model.student_model Qwen/Qwen2.5-Omni-7B   --model.teacher_model Qwen/Qwen2.5-Omni-7B   --data.train_data data/train.jsonl   --data.val_data data/val.jsonl   --data.output_dir outputs/echodistill
```

## Notes before public release

- Add your final license before publishing.
- Verify that the underlying audio dataset license permits redistribution or link to its original source.
- The included code keeps compatibility patches for Qwen2.5-Omni, PEFT/QLoRA, multi-GPU dispatch,
  and audio loading fallbacks.
