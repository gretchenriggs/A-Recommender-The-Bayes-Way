import pandas as pd
import numpy as np
import pickle
import spacy
import re

class Preprocessor:
    '''
    Class which transforms input json data points into X inputs for the model.
    Takes a few seconds to initialize, needs to load spacy/nlp.
    '''

    def __init__(self):
        self.nlp = spacy.load('en')

    def transform(self, json_data):
        df = pd.DataFrame([json_data])
        explode(df)
        average = df['sale_duration'].mean()
        X_df = self.make_train_set(df, average)
        X_df.sort_index(axis=1, inplace=True)
        return X_df

    # function to make the training set
    def make_train_set(self, entry, avg, one_point = False):

        '''
        entry = the input data frame or point
        avg = average of mean of the sale_duration col
        one_point = False is when entry is a full frame True is for a single point

        '''

    # ------------ columns of the original data frame to be used later

        cols = [u'acct_type',
         u'approx_payout_date',
         u'body_length',
         u'channels',
         u'country',
         u'currency',
         u'delivery_method',
         u'description',
         u'email_domain',
         u'event_created',
         u'event_end',
         u'event_published',
         u'event_start',
         u'fb_published',
         u'gts',
         u'has_analytics',
         u'has_header',
         u'has_logo',
         u'listed',
         u'name',
         u'name_length',
         u'num_order',
         u'num_payouts',
         u'object_id',
         u'org_desc',
         u'org_facebook',
         u'org_name',
         u'org_twitter',
         u'payee_name',
         u'payout_type',
         u'previous_payouts',
         u'sale_duration',
         u'sale_duration2',
         u'show_map',
         u'ticket_types',
         u'user_age',
         u'user_created',
         u'user_type',
         u'venue_address',
         u'venue_country',
         u'venue_latitude',
         u'venue_longitude',
         u'venue_name',
         u'venue_state']


    # ---------------- checking if the data is one line or the whole train
    # may need to be edited later

        if one_point:
            entry = np.array(entry).reshape(1,44)
            df = pd.DataFrame(entry)
            df.columns = cols

        else:
            df = entry


    # --------- labels the columns into categories for EDA
        # the label col
        y = ['acct_type']

        # cols with date
        date = ['approx_payout_date',
            'event_created',
            'event_end',
            'event_start',
            'user_created']

        # number colums
        num = ['body_length',
                'channels',
                'delivery_method',
                'gts',
                'num_payouts',
                'name_length',
                'org_facebook',
                'org_twitter',
                'sale_duration',
                'sale_duration2',
                'user_age']

        # categorical colums
        cat = ['fb_published',
                'has_analytics',
                'has_header',
                'has_logo',
                'listed',
                'show_map']


        # create dummies
        to_dummy = ['delivery_method',
                'payout_type',#has blanks
                'sale_duration',
                'user_type' # has strange values
                ]

        # columns to drop
        to_drop = ['num_order','object_id','currency']

        # colums to explode and pull out internal information
        explode = ['previous_payouts','ticket_types']

        # to turn it into
        has_value = ['country',
             'email_domain',
             'name',
             'org_name',
             'payee_name',
             'venue_address',
             'venue_country',
             'venue_name',
             'venue_state',
             'event_published',
             'org_facebook',
             'org_twitter',
             'venue_latitude',
             'venue_longitude',
             'description',
             'org_desc']

        nlp_cols = [
             'description',
             'org_desc']

    # --------- updating the columns by category

        # cateforical columns
        df['delivery_method'] = df['delivery_method'].map(lambda x: sub_na(x,4.))
        df['sale_duration'] = df['sale_duration'].map(lambda x: sub_na(x,avg))


        # categorical columns that need individiual change
        df['has_header'] = df['has_header'].map(lambda x: sub_na(x,0))
        df['listed'] = df['listed'].map(yes_no)

        # lat and long features
        df['lat'] = df['venue_latitude'].map(lambda x: sub_na(x,0))
        df['long'] = df['venue_longitude'].map(lambda x: sub_na(x,0))

        # add nlp columns
        # nlp = spacy.load('en')
        for col in nlp_cols:
            vec = compute_nlp_vectors(self.nlp, df, col)
            add_nlp(df, vec, col_name='{}_nlp'.format(col))

        # creates has variables
        for col in has_value:
            df[col] = df[col].fillna('')
            name = 'has_{}'.format(col)
            df[name] = df[col].map(get_has_value)
            df = df.drop(col,axis = 1)

        # does the sales differnce and has negative
        df['sale_dur_diff'] = abs(df['sale_duration'] - df['sale_duration2'])
        df['sale_dur_diff'].map(lambda x: sub_na(x,0))
        df['has_neg_sale_dur'] = df['sale_duration'].map(is_neg)

        # creates dummies for payment type
        check = []
        ach = []
        pay_other = []
        for x in df['payout_type']:
            if x == 'CHECK':
                check.append(1)
                ach.append(0)
                pay_other.append(0)
            elif x == 'ACH':
                check.append(0)
                ach.append(1)
                pay_other.append(0)
            else:
                check.append(0)
                ach.append(0)
                pay_other.append(1)
        df['pay_check'] = check
        df['pay_ach'] = ach
        df['pay_other'] = pay_other

        # dropping other cols
        drop_lst = to_drop + date + explode +['payout_type']
        df = df.drop(drop_lst, axis = 1)

        return df


# functions to later map to columns


# function to make has classifier columns: if value is nan return 0
def get_has_value(x):
    if x == None or x == u'': #or np.isnan(x)== True:
        return 0
    else:
        return 1

# changes yes / no to 1 / 0
def yes_no(x):
    if x == 'y':
        return 1
    else:
        return 0

# returns 1 if negative 0 if positive
def is_neg(x):
    if x < 0:
        return 1
    else:
        return 0

# cutome sub nan values with input value made to ensure it could be used with a single point
def sub_na(x,sub_value):
    if x is None:
        return sub_value
    elif np.isnan(x):
        return sub_value
    else:
        return x





# ------------------------------------------------------------
def explode(df):
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
            # print "i: {0}, len_dict: {1}".format(i, len_dict)
            cost += df['ticket_types'][indx][i]['cost']
            # print "cost: ", cost
            quantity_sold += df['ticket_types'][indx][i]['quantity_sold']
            # print "quantity_sold", quantity_sold
            quantity_total += df['ticket_types'][indx][i]['quantity_total']
            # print "quantity_total", quantity_total
            event_id.append(df['ticket_types'][indx][i]['event_id'])
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
    # df['old_index'] = df.index
    # df['new_index'] = range(len(df))
    # df.set_index('new_index', inplace=True)

    # Adding aggregated cost, quantity_sold, quantity_total, and
    #   event_id columns
    df['cost'] = pd.DataFrame((np.array(all_cost)).T)
    df['quantity_sold'] = pd.DataFrame((np.array(all_quant_sold)).T)
    df['quantity_total'] = pd.DataFrame((np.array(all_quant_total)).T)
    # df['event_id'] = pd.DataFrame((np.array(all_event_id)).T)

def compute_nlp_vectors(nlp, df, column='description'):
    description_vectors = []
    for i in df.index:
        description = df.loc[i,column]
        if type(description) == unicode:
            description = clean_html(description)
            if len(description) > 0:
                description_nlp = nlp(description)
                description_vector = description_nlp.vector
            else:
                description_vector = np.zeros(300)
        else:
            description_vector = np.zeros(300)
        description_vectors.append(description_vector)
    description_vectors = np.array(description_vectors)
    return description_vectors

    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    X_json = r.json()
    X_input = preprocessor.transform(X_json)
    y = model.predict(X_input)

def clean_html(raw_html):
    '''
    Removes all html tags from a string.
    Taken from here: http://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def add_nlp(df,nlp, col_name = 'nlp'):
    for n in range(nlp.shape[1]):
        name = '{}_{}'.format(col_name,n)
        df[name] = nlp[:,n]

if __name__ == '__main__':
    df_all = pd.read_json('../data/data.json')
    df = df_all.copy()

    preprocessor = Preprocessor()
    explode(df)
    average = df_all['sale_duration'].mean()
    df = preprocessor.make_train_set(df, avg = average)
    df.sort_index(axis=1, inplace=True)

    # adding nlp spacy magic to add 300 features
    # nlp = pickle.load(open('description_vectors.pkl'))
    # add_nlp(df,nlp)
    # org = pickle.load(open('org_desc_vectors.pkl'))
    # add_nlp(df,org,col_name = 'org')

    # test point changes some to objects...
    # test_point = df_all.iloc[3,:].values
    # test = make_train_set(test_point, avg = average,one_point = True)

    # to make a
    df.to_csv('clean_df_5.csv')
