# Quick Start

## Install Dependencies

```bash
pip install torch==2.1.2 transformers==4.33.0 datasets tiktoken numpy==1.26.4 wandb
```

## Data Preparation

Prepare the [OpenWebText](https://huggingface.co/datasets/openwebtext) data as:

```bash
python data/openwebtext/prepare.py
```

## **Start Training**

```bash
bash run.sh
```

You can also easily use other optimizers by changing "optimizer_name" in the config file.

```python
# For Adam
optimizer_name = "adam"
# For Muon
optimizer_name = "muon"
```

## Acknowledgements

This repo is built upon [nanoGPT](https://github.com/karpathy/nanoGPT/), we thank the authors for their great work.
