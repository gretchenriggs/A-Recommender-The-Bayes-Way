import pandas as pd
import numpy as np
import cPickle as pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def update_all_df(df):
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
    for indx in df.index:
        len_dict = len(df['ticket_types'][indx])
        trans_num.append(len_dict)
        cost = 0.0
        quantity_sold = 0
        quantity_total = 0
        event_id = []
        for i in xrange(len_dict):
            print "i: {0}, len_dict: {1}".format(i, len_dict)
            cost += df['ticket_types'][indx][i]['cost']
            print "cost: ", cost
            quantity_sold += df['ticket_types'][indx][i]['quantity_sold']
            print "quantity_sold", quantity_sold
            quantity_total += df['ticket_types'][indx][i]['quantity_total']
            print "quantity_total", quantity_total
            event_id.append(df['ticket_types'][indx][i]['event_id'])
        if event_id != []:
            event_id = list(set(event_id))[0]
        else:
            event_id = None
        all_cost.append(cost)
        all_quant_sold.append(quantity_sold)
        all_quant_total.append(quantity_total)
        all_event_id.append(event_id)

    # Adding aggregated cost, quantity_sold, quantity_total, and
    df['cost'] = pd.DataFrame((np.array(all_cost)).T)
    df['quantity_sold'] = pd.DataFrame((np.array(all_quant_sold)).T)
    df['quantity_total'] = pd.DataFrame((np.array(all_quant_total)).T)
    df['event_id'] = pd.DataFrame((np.array(all_event_id)).T)

    # Creating currency exchange rate to USD and creating column with the
    #   cost in USD for each row
    curr_exchange = []
    rate = 0.00
    for x in xrange(len(df)):
        print df['currency'][x]
        if df['currency'][x] == 'AUD':
            rate = 0.77
        if df['currency'][x] == 'CAD':
            rate = 0.76
        if df['currency'][x] == 'EUR':
            rate = 1.06
        if df['currency'][x] == 'GBP':
            rate = 1.25
        if df['currency'][x] == 'MXN':
            rate = 0.049
        if df['currency'][x] == 'USD':
            rate = 1.00
        # else:
        # #     rate = None
        # print rate
        curr_exchange.append(rate)
    df['exchange_rate_to_USD'] = pd.DataFrame(np.array(curr_exchange))
    df['cost_in_USD'] = df['exchange_rate_to_USD'] * df['cost']

    # Adding fraud column, 1 if acct_type begins with fraud, 0 otherwise
    fraud_events = ['fraudster', 'fraudster_event', 'fraudster_att']
    df['fraud'] = df['acct_type'].isin(fraud_events).astype(int)
    return df

# Computing confusion matrix, formatting in standard setup
def standard_confusion_matrix(y_true, y_predict):
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_predict)
    return np.array([[tp, fp], [fn, tn]])

# Create the profit curve to be plotted for analysis
def run_profit_curve(model, costbenefit, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    thresholds = sorted(probabilities, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = probabilities > threshold
        confusion_mat = standard_confusion_matrix(y, y_predict)
        profit = np.sum(confusion_mat * costbenefit) / float(len(y))
        profits.append(profit)
    return thresholds, profits

def plot_profit_models(model, costbenefit, X, y):
    percentages = np.linspace(0, 100, len(y))
    thresholds, profits = run_profit_curve(model, costbenefit, X, y)
    plt.plot(percentages, profits, label="Random Forest")
    plt.title("Profit Curve - Maximum Profit Identifier")
    plt.xlabel("Percentage of instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.xlim(0,15)
    plt.legend(loc='best')
    plt.savefig('profit_curve.png')
    plt.show()

if __name__ == '__main__':

    # Read in data from json file
    df = pd.read_json("data/data.json")

    # Clean up df and add columns
    df = update_all_df(df)

    # Computing cost_per_event & quant_total_per_event for non-fraud events
    fraud_events = ['fraudster', 'fraudster_event', 'fraudster_att']
    cost_per_event = df[~df['acct_type'].isin(fraud_events)].groupby('event_id').sum()['cost']
    quant_tot_per_event =  df[~df['acct_type'].isin(fraud_events)].groupby('event_id')\
                         .sum()['quantity_total']

    # Reading in clean data used in Model Building and Random Forest model
    with open("data/random_forest.pkl") as f_un:
        model = pickle.load(f_un)
    df2 = pd.read_csv('data/clean_df_4.csv')

    # X is 644 columns in df2 input to create Random Forest model
    # y comes from the df fraud column
    df2.pop('acct_type')
    X = df2
    y = df['fraud']

    # Predicting y values based on Random Forest model
    y_predict = model.predict(X)

    # Need to get confusion_mat
    costbenefit = np.array([[-450, -50], [0, 0]])
    confusion_mat = standard_confusion_matrix(y, y_predict)
    profit = np.sum(confusion_mat * costbenefit) / float(len(y))
    plot_profit_models(model, costbenefit, X, y)
