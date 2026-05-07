# Dataset Card

This release contains JSONL files for EchoDistill-style noisy-to-clean audio distillation.
Each row stores only the static information needed to reproduce training:

- `id`: sample identifier.
- `prompt`: task prompt shown to the model.
- `noisy_audio_path`: relative path to the noisy student audio.
- `clean_audio_path`: relative path to the paired clean teacher audio.
- `choices`: candidate answer choices when available.
- `target`: target answer.
- `noise_type`: noise category.
- `snr`: signal-to-noise ratio.

Candidate responses and rewards are not static annotations. During training, the student samples
candidate responses online and the trainer computes rewards from the sampled response, target,
and choices.

Audio files are not bundled in this package. Place the original audio assets under your dataset
root and use `data.path_maps` or `--data.path_map` to map the relative paths in the JSONL files to
your local audio directory.
