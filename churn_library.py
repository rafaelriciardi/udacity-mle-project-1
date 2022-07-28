# library doc string
'''
This library contains all the code and functions needed to run the training pipeline

Author: Rafael Barreira
Date: 27/07/2022
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report, auc, RocCurveDisplay, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import os
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    original_df = pd.read_csv(pth)
    return original_df


def perform_eda(original_df, verbose=False):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    if verbose:
        print('Dataframe shape:', original_df.shape, '\n')

        print('Dataframe Null values:')
        print(original_df.isnull().sum(), '\n')

        print('Dataframe description:')
        print(original_df.describe(), '\n')

    original_df['Churn'] = original_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    churn_hist_plot = original_df['Churn'].hist()
    plt.savefig('images/eda/churn_histogram.png')

    cust_age_hist_plot = original_df['Customer_Age'].hist()
    plt.savefig('images/eda/cust_age_histogram.png')

    marital_status_count_plot = original_df.Marital_Status.value_counts(
        'normalize').plot(kind='bar')
    plt.savefig('images/eda/marital_status_count.png')

    total_trans_dist_plot = sns.distplot(original_df['Total_Trans_Ct'])
    plt.savefig('images/eda/total_trans_dist_plot.png')

    corr_matrix = sns.heatmap(
        original_df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig('images/eda/corr_matrix.png')
    plt.close()


def encoder_helper(encoding_df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for column in category_lst:
        column_values_lst = []
        column_groups = encoding_df.groupby(column).mean()[response]

        for val in encoding_df[column]:
            column_values_lst.append(column_groups.loc[val])

        new_column_name = column + '_' + response
        encoding_df[new_column_name] = column_values_lst

    return encoding_df


def perform_feature_engineering(df, response='Churn'):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_' + response,
        'Education_Level_' + response,
        'Marital_Status_' + response,
        'Income_Category_' + response,
        'Card_Category_' + response]

    y_original = df[response]
    x__original = df[keep_cols]

    x_train, x_test, y_train, y_test = train_test_split(
        x__original, y_original, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/rf_results.png', bbox_inches='tight')

    plt.clf()
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('images/results/lr_results.png', bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth, bbox_inches='tight')
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    plt.savefig('images/results/roc_auc_curve.png', bbox_inches='tight')
    plt.close()

    feature_importance_plot(
        cv_rfc,
        X_test,
        'images/results/feature_importance.png')


if __name__ == "__main__":
    data = import_data("./data/bank_data.csv")
    perform_eda(data)

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    encoded_data = encoder_helper(data, cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        encoded_data)
    train_models(X_train, X_test, y_train, y_test)
