## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --config configs/default.yaml
```

## Resume
```bash
python -m src.train --config configs/default.yaml --resume outputs/best.pt
```
