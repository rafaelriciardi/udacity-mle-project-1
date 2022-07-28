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

├── data<br />
│   └── bank_data.csv<br />
├── images<br />
│   ├── eda<br />
│   │   ├── churn_histogram.png<br />
│   │   ├── corr_matrix.png<br />
│   │   ├── cust_age_histogram.png<br />
│   │   ├── marital_status_count.png<br />
│   │   └── total_trans_dist_plot.png<br />
│   └── results<br />
│       ├── feature_importance.png<br />
│       ├── lr_results.png<br />
│       ├── rf_results.png<br />
│       └── roc_auc_curve.png<br />
├── logs<br />
│   └── churn_library.log<br />
├── models<br />
│   ├── logistic_model.pkl<br />
│   └── rfc_model.pkl<br />
├── churn_library.py<br />
├── churn_notebook.ipynb<br />
├── churn_script_logging_and_tests.py<br />
├── README.md<br />
└── requirements.txt<br />
