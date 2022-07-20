# BexAI

BexAI is a benchmark suite to empower developers and architects to evaluate new and existing AI models and explainers using a wide range of datasets and metrics. In its current implementation it evaluate three model agnostic black box model explainers - LIME, SHAP and Anchor.

# Website

If you want to know more about the project or find description about resources used, please check our [website](https://adacenter.org/bexai/)

## Evaluation metrics

Explainers are evaluated based on the following two quantitative attributes.

**Runtime**: is the time taken to exlain a single instance from the test data.

**Fidelity**: is measured by first by removing each feature by replacing it with its mean value and get prediction for this instance. Then measure the correlation between its importance and the change in prediction. Currently, fidelity metrics are implmemented for tabular datastes

## Installation

Install the required dependencies in requirement.txt using the following command

```bash
python3 -m venv env 
source env/bin/activate
pip3 install -r requirements.txt
```

Download the datasets zip file from [here](https://drive.google.com/file/d/1YBCa4VltDhoXxhOmrRw-35RDT7vnhyyC/view?usp=sharing). After extracting it, inlcude the datasets folder in the main directory of the project.

## Usage

Use the following command to evaluate the explainers based on a selected dataset

dataset_name can be titanic,cancer,iris,wine,diabetes,loan,reddit,mnist

```bash
python3 main.py --dataset <dataset_name>
```

only one evaluation can be selected and run for a dataset using the evaluation optional argument. evaluation_metric can be runtime, fidelity.

```bash
python3 main.py --dataset <dataset_name> --evaluation <evaluation_metric>
```

A graphical explanation can also be generated for a prediction on a row of the test data. The outputs are placed in explanation/<dataset_name> directory.

```bash
python3 main.py --dataset <dataset_name> --explain <index_on_x_test>
```
