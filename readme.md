## Steps to run

# BexAI

BexAI is a benchmark suite to empower developers and architects to evaluate new and existing AI models and explainers using a wide range of datasets and metrics. In its current implementation it evaluate three model agnostic black box model explainers - LIME, SHAP and Anchor.

## Evaluation metrics

**Runtime**: is the time taken to exlain a single instance from the test data.

**Fidelity**: is measured by first by removing each feature by replacing it with its mean value and get prediction for this instance. Then measure the correlation between its importance and the change in prediction. Currently, fidelity metrics are implmemented for tabular datastes

## Installation

Install the required dependencies in requirement.txt using the following command

```bash
python3 -m venv bin #create a virutal environment
source env/bin/activate
pip install -r requirements.txt
```

## Usage

Use the following command to evaluate the explainers based on a selected dataset

```bash
python3 main.py --dataset <dataset_name>
```

only one evaluation can be selected and run for a dataset using the evaluation optional argument

```bash
python3 main.py --dataset <dataset_name> --evaluation <evaluation_metric>
```

A graphical explanation can also be generated for a prediction on a row of the test data. The outputs are placed in explanation/<dataset_name> directory.

```bash
python3 main.py --dataset <dataset_name> --explain <index_on_x_test>
```

python3 -m venv bin #create a virutal environment
source env/bin/activate
pip3 install -r requirements.txt
python3 main.py
