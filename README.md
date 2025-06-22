# GPU-Accelerated Deep Learning Model Benchmark

## Objective
Compare CPU vs GPU training performance using PyTorch and CIFAR-10 dataset.

## Structure

- `src/` — Source code (model, training, benchmarking, utils)
- `data/` — Dataset storage
- `models/` — Saved models
- `results/` — Benchmarking results
- `tests/` — Unit tests
- `.github/workflows/` — CI pipeline

deep-learning-benchmark/
├── data/
├── models/
├── results/
├── src/
│   ├── __init__.py
│   ├── resnet_module.py
│   ├── train.py
│   ├── benchmark.py
│   └── utils.py
├── tests/
│   ├── __init__.py
│   └── test_resnet.py
├── requirements.txt
├── .gitignore
├── pytest.ini
├── README.md
└── .github/
    └── workflows/
        └── ci.yml

## Usage

1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run training and benchmarking:
    ```bash
    python -m src.benchmark
    ```
3. Run tests:
    ```bash
    python -m pytest
    ```


## CI/CD

GitHub Actions runs unit tests on every push.