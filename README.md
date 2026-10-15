# Soybean Pod Detection Dataset and Code

A field-grown soybean pod detection dataset with bounding-box annotations, collected from 113 diverse soybean accessions, together with training/evaluation code and model weights.

## Dataset

The dataset comprises 1,130 RGB images from 113 soybean accessions (10 plants per accession), grown in 2025 at the Chongqing Sub-center for Plant Variety Testing, Ministry of Agriculture and Rural Affairs, China. The accessions span diverse plant architectures (erect, semi-erect, and semi-prostrate; plant height 20--130 cm) and pod densities (sparse to dense). The annotated class is `pod`.

| Split | Images | Ratio |
|-------|--------|-------|
| train | 780    | 70%   |
| val   | 222    | 20%   |
| test  | 111    | 10%   |

Images and labels share the same filename under `images/` and `labels/` respectively.

## Independent Test Set

A separate 200-image test set from ~20 previously unseen accessions is available in [Release v1.1](../../releases/tag/v1.1), organized by pod density:

| Density | Images |
|---------|--------|
| sparse  | 60     |
| medium  | 80     |
| dense   | 60     |

## Code

Training and evaluation code is in the `code/` directory:

| File | Description |
|------|-------------|
| `train.py` | RT-DETR training script with mAP@0.5 best checkpoint saving |
| `evaluate.py` | Inference and evaluation with HSV pre-cropping and DAC post-processing |
| `requirements.txt` | Python dependencies |

### Quick Start

```bash
cd code
pip install -r requirements.txt

# Training
python train.py --model-cfg rtdetr-l.yaml --weights rtdetr-l.pt --data data.yaml

# Evaluation on independent test set
python evaluate.py \
  --img-dirs ../independent_test/sparse ../independent_test/medium ../independent_test/dense \
  --model best.pt
```

See `code/README.md` for full usage instructions.

## Model Weights

The trained checkpoint (`best.pt`) is available in [Release v1.1](../../releases/tag/v1.1)
