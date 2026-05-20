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

- **NTIRE 2020** (ARAD, spectral reconstruction from RGB): challenge code & data instructions at
  [boazarad/NTIRE2020_spectral](https://github.com/boazarad/NTIRE2020_spectral)
  (data via the CodaLab competition linked there).
- **NTIRE 2022** (ARAD_1K): [boazarad/ARAD_1K](https://github.com/boazarad/ARAD_1K) ·
  CodaLab: <https://codalab.lisn.upsaclay.fr/competitions/721>
- **Hyper-Skin**: [hyperspectral-skin/Hyper-Skin-2023](https://github.com/hyperspectral-skin/Hyper-Skin-2023)
  (request access via the EULA in the repo).
- **Choledoch** (downstream): [Kaggle — Multidimensional Choledoch Database](https://www.kaggle.com/datasets/ethelzq/multidimensional-choledoch-database)
- **HeiPorSPECTRAL** (downstream): <https://www.heiporspectral.org/> ·
  Zenodo: <https://doi.org/10.5281/zenodo.7737674>

Expected structure:

```
hsiData/
├── ntire2020/
├── ntire2022/
└── hyperskin/
```
