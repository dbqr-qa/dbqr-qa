# DBQR-QA
A Question Answering Dataset on a Hybrid of Database Querying and Reasoning

## Introduction
This Python package includes tools that help read the dataset and evaluate the results. The package also contains built-in functions (optional) for programs in the annotated labels.

For more information, visit:
- The project's website: https://dbqr-qa.github.io
- Quick start: https://dbqr-qa.github.io/quickstart.html
- Downloads: https://dbqr-qa.github.io/downloads.html
- Paper: https://aclanthology.org/2024.findings-acl.900/

## Setup
To install the package run:
```
pip install dbqrqa
```

To build and install the package locally, run:
```
python -m build
pip install dist/dbqrqa-[version].tar.gz
```

## Package Structure
Package Structure
1. dataset.py: Read and prepare the dataset for training/prediction
2. evaluation.py: Heuristic and GPT evaluators
3. builtins.py: Built-in functions for program annotation

## Unit Test
For the unit test, run the following command:
```
python -m unittest
```

The unit-test command does not test the GPT evaluator due to cost and security considerations. To test the GPT evaluator, obtain an OpenAI's API key, then run the following command:
```
python tests/gpt_evaluation.py 
```

## Citation
Use the following BibTex or get the citation in other formats from [ACL Anthology](https://aclanthology.org/2024.findings-acl.900/):
```
@inproceedings{nararatwong-etal-2024-dbqr,
    title = "{DBQR}-{QA}: A Question Answering Dataset on a Hybrid of Database Querying and Reasoning",
    author = "Nararatwong, Rungsiman  and
      Chen, Chung-Chi  and
      Kertkeidkachorn, Natthawut  and
      Takamura, Hiroya  and
      Ichise, Ryutaro",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.900",
    pages = "15169--15182"
}
```
