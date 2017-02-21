import pandas as pd
import numpy as np

def fraud_column(df):
    '''
    Add column called fraud based on acct_type column.

    Input: initial dataframe without labeled fraud

    Returns: dataframe with label of fraud (1) or not fraud (0)
    '''

    df['fraud'] = df.apply(label_fraud, axis=1)
    df_final = df.drop('acct_type', axis=1)
    return df_final

def label_fraud(row):
    '''
    Label one row of fraud dataframe at a time. This is easy to make more generalized by adding elifs to the function.

    Input: Unlabeled row in the fraud dataframe

    Returns: Labeled row. Fraud is labeled if the account type is 'fraudster_event', 'fraudster', 'fraudster_att'
    '''

    # fraud_labels are row values in the account type column that will result in the row being labeled fraud (1)
    fraud_labels = ['fraudster_event', 'fraudster', 'fraudster_att']

    if row['acct_type'] in fraud_labels:
        val = 1
    else:
        val = 0
    return val
