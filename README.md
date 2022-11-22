# Banca Massiccia: Credit Risk Modeling

## Model Evaluation
This repository contains two files, `poetry.lock` and `pyproject.toml`, which encode the dependencies needed for our corporate default model to run.

Open a terminal and `cd` into the folder of the cloned repo. Then, from this folder, run the following commands in the terminal to set up the necessary Anaconda virtual environment:
```
conda create -n CDSML poetry
conda activate CDSML
poetry install
```

Once the virtual environment has been set up, run the following command. This will pass your specified input data to the model, and will produce a CSV file `predictions.csv`, containing predicted default probabilities for each company.
```
python model_evaluate.py <your_input_data>.csv
```

## Appendix

The plots and charts shown in the slide deck come from the Jupyter notebooks `default.ipynb` and `EDA.ipynb`.