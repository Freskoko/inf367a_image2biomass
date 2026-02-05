# INF367A

Kaggle: [Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass)

## Requirements

See pyproject.toml. The environment manager [uv](https://docs.astral.sh/uv/getting-started/installation/) is ready to use in this repo.

pip install uv
uv --version
uv venv .venv ### Create a venv  
uv sync ### Install exactly what’s in uv.lock
. venv\Scripts\activate       


## Formatting

Henrik likes to use [ruff](https://docs.astral.sh/ruff/) to format code

## How to run

Run [predict.py](/src/main/predict.py)

cd src  
python -m main.run_only_train
