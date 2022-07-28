# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description


In this project, you will implement your learnings to identify credit card customers that are most likely to churn. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).


## Running Files

Before running the files, you need to install the correct packages for this project. To achieve this, you should run the following command:
```
pip install -r requirements.txt
```

Then, to ensure everything is working properly and the pipeline is doing what it is suposed to, you can run the tests:
```
python churn_script_logging_and_test.py
```

Finally, to run the hole pipeline and train the model:
```
python churn_library.py
```

## Files in the repo

Below theres a representation of the repos structure.

├── data
│   └── bank_data.csv
├── images
│   ├── eda
│   │   ├── churn_histogram.png
│   │   ├── corr_matrix.png
│   │   ├── cust_age_histogram.png
│   │   ├── marital_status_count.png
│   │   └── total_trans_dist_plot.png
│   └── results
│       ├── feature_importance.png
│       ├── lr_results.png
│       ├── rf_results.png
│       └── roc_auc_curve.png
├── logs
│   └── churn_library.log
├── models
│   ├── logistic_model.pkl
│   └── rfc_model.pkl
├── churn_library.py
├── churn_notebook.ipynb
├── churn_script_logging_and_tests.py
├── README.md
└── requirements.txt
