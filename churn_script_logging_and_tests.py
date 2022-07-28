'''
This script is responsible for testing all the functions needed to run the training pipeline

Author: Rafael Barreira
Date: 27/07/2022
'''


import os
import logging
import churn_library as cls

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing perform_eda: An error ocurred during the EDA step")
        raise e

    try:
        assert os.path.isfile('images/eda/churn_histogram.png')
        assert os.path.isfile('images/eda/cust_age_histogram.png')
        assert os.path.isfile('images/eda/marital_status_count.png')
        assert os.path.isfile('images/eda/total_trans_dist_plot.png')
        assert os.path.isfile('images/eda/corr_matrix.png')
        logging.info("Testing perform_eda: Files saved successfully")
    except AssertionError as err:
        logging.error("Testing perform_eda: Files not saved successfully")
        raise err


def test_encoder_helper(encoder_helper, df, category_lst):
    '''
    test encoder helper
    '''

    try:
        original_df_size = len(df.columns)
        df_encoded = encoder_helper(df, category_lst)
        logging.info("Testing encoder_helper: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing encoder_helper: An error ocurred during the encoding step")
        raise e

    try:
        assert df_encoded.shape[0] > 0
        assert df_encoded.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The encoded file generated doesn't appear to have rows and columns")
        raise err

    try:
        assert len(df_encoded.columns) == original_df_size + len(category_lst)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Expected different amount of columns after encoding")
        raise err

    return df_encoded


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing perform_feature_engineering: An error ocurred during the feature engineering step")
        raise e

    try:
        assert len(X_train) == len(y_train)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Different amount of rows in X_train and y_train")
        raise err

    try:
        assert len(X_test) == len(y_test)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Different amount of rows in X_test and y_test")
        raise err

    try:
        assert len(X_train.columns) == len(X_test.columns)
        assert len(X_train.columns) == 19
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Expected different amount of columns after feature engineering")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing test_train_models: SUCCESS")
    except Exception as e:
        logging.error(
            "Testing test_train_models: An error ocurred during the training step")
        raise e

    try:
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: Models not saved successfully")
        raise err

    try:
        assert os.path.isfile('images/results/roc_auc_curve.png')
        assert os.path.isfile('images/results/feature_importance.png')
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: Reports not saved successfully")
        raise err


if __name__ == "__main__":
    df = test_import(cls.import_data)
    test_eda(cls.perform_eda, df)
    df_encoded = test_encoder_helper(cls.encoder_helper, df, cat_columns)
    X_train, X_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, df_encoded)
    test_train_models(cls.train_models, X_train, X_test, y_train, y_test)
