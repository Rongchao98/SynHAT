# SynHAT: A Two-stage Coarse-to-Fine Diffusion Framework for Synthesizing Human Activity Traces 

Offical code for "SynHAT: A Two-stage Coarse-to-Fine Diffusion Framework for Synthesizing Human Activity Traces". 

## Layout
- `train_s1.py`, `train_s2.py`: training entry points (default to `config/sample_synhat.yaml`).
- `inference_s1.py`, `inference_s2.py`, `inference_s3.py`: sampling scripts for the three stages.
- `model/`, `utils/`: lightweight modules required by the scripts; no evaluation helpers are included.
- `data/sample_synhat/`: synthetic processed dataset + metadata/POI catalog used by the sample config.
- `scripts/generate_sample_dataset.py`: recreates the synthetic dataset.

## Environment
Install required packages through:
```bash
pip install torch torchvision torchaudio numpy pyyaml
```

## Quick Start
1. (Optional) Regenerate sample data
   ```bash
   python scripts/generate_sample_dataset.py
   ```
2. Train Coarse-HADiff in Stage-1 
   ```bash
   python train_s1.py --config config/sample_synhat.yaml --output-dir outputs/stage1_demo
   ```
3. Train Fine-HADiff in Stage-2
   ```bash
   python train_s2.py --config config/sample_synhat.yaml --output-dir outputs/stage2_demo
   ```
4. Run inference once checkpoints are available
   - Stage-1: `python inference_s1.py --config config/sample_synhat.yaml --checkpoint <path>`
   - Stage-2: `python inference_s2.py --config config/sample_synhat.yaml --checkpoint <path> --stage1-samples <stage1_samples.npz>`
   - Stage-3: `python inference_s3.py --config config/sample_synhat.yaml --stage2-events data/sample_synhat/stage2_events_sample.npz`

