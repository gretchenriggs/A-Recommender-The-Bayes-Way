import pandas as pd
import numpy as np

def update_df_fraud(df_fraud):
    # Initializing: Number of transactions for event_id in ticket_types
    #               Total cost of all these transactions per user
    #               Total quantity sold in these transactions
    #               Total quantity_total in these transactions
    #               All event_ids, which is one per each transaction
    trans_num = []
    all_cost = []
    all_quant_sold = []
    all_quant_total = []
    all_event_id = []

    # Looping through and summing the cost, quantity_sold, and
    #   quantity_total for each transaction per event_id
    # Making list of event_ids
    for indx in df_fraud.index:
        len_dict = len(df_fraud['ticket_types'][indx])
        trans_num.append(len_dict)
        cost = 0.0
        quantity_sold = 0
        quantity_total = 0
        event_id = []
        for i in xrange(len_dict):
            print "i: {0}, len_dict: {1}".format(i, len_dict)
            cost += df_fraud['ticket_types'][indx][i]['cost']
            print "cost: ", cost
            quantity_sold += df_fraud['ticket_types'][indx][i]['quantity_sold']
            print "quantity_sold", quantity_sold
            quantity_total += df_fraud['ticket_types'][indx][i]['quantity_total']
            print "quantity_total", quantity_total
            event_id.append(df_fraud['ticket_types'][indx][i]['event_id'])
        if event_id != []:
            event_id = list(set(event_id))[0]
        else:
            event_id = None
        all_cost.append(cost)
        all_quant_sold.append(quantity_sold)
        all_quant_total.append(quantity_total)
        all_event_id.append(event_id)

    # Saving out old index used in original DataFrame containing all
    #   transaction activity (both fraudulent and non-fraudulent).
    # Reindexing the df_fraud Dataframe to consecutive integers so can
    #   easily add aggregated cost, quantity_sold, quantity_total, and
    #   event_id columns
    df_fraud['old_index'] = df_fraud.index
    df_fraud['new_index'] = range(len(df_fraud))
    df_fraud.set_index('new_index', inplace=True)

    # Adding aggregated cost, quantity_sold, quantity_total, and
    #   event_id columns
    df_fraud['cost'] = pd.DataFrame((np.array(all_cost)).T)
    df_fraud['quantity_sold'] = pd.DataFrame((np.array(all_quant_sold)).T)
    df_fraud['quantity_total'] = pd.DataFrame((np.array(all_quant_total)).T)
    df_fraud['event_id'] = pd.DataFrame((np.array(all_event_id)).T)

    return df_fraud


if __name__ == '__main__':

    # Read in data from json file
    df = pd.read_json("data/data.json")

    # Classify fraud_events as those 'acct_types' starting with 'fraud'
    #  and siphoning into new DataFrame for analysis
    fraud_events = ['fraudster', 'fraudster_event', 'fraudster_att']
    df_fraud = df[df['acct_type'].isin(fraud_events)]

    # Clean up and add columns to df_fraud
    df_fraud = update_df_fraud(df_fraud)
