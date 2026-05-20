# PHASE — Physiology-Aware Hyperspectral Reconstruction

Object-to-human hyperspectral (HSI) reconstruction via semi-supervised domain
adaptation, built on a Mean-Teacher with two
physiological components:

- **PCR** (`helpers/pcr.py`) — spectral-density-guided channel-adaptive masking on
  the student view; spectrally dominant bands are masked more to break source
  channel shortcuts.
- **PCA** (`helpers/pca.py`) — dual prototype banks with a reliability-gated alignment loss.

Objective: `L = L_sup + λ_un(t) · (L_con + λ_pca · L_PCA)`.

## Run

```bash
pip install -r requirements.txt
python train_phase.py --data_dir /path/to/data --num_shot 10   # 5% labeled; --num_shot 3 for 1.5%
```

## Layout

```
train_phase.py                          # SSDA training entry
helpers/{pcr,pca}.py                    # PCR masking, PCA dual-bank alignment
helpers/{utils,metrics}.py              # logging, SSIM/SAM/PSNR
hsiData/                                # NTIRE / Hyper-Skin datasets
models/reconstruction/MST_Plus_Plus.py  # backbone
```

## Datasets

Download the datasets and place them under `hsiData/`:

- **NTIRE 2020**: [download link]
- **NTIRE 2022**: [download link]
- **Hyper-Skin**: [download link]
- **Choledoch / HeiPorSPECTRAL** (downstream): [download links]

Expected structure:

```
hsiData/
├── ntire2020/
├── ntire2022/
└── hyperskin/
```
