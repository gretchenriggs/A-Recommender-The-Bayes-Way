import pandas as pd
import numpy as np
from taylor_eda import fraud_column
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import matplotlib.pylab as plt
import pickle

def preprocess(df):
    '''
    Get the data ready to build a model. Take a pandas dataframe and creates a train test split and makes numpy arrays that are ready to be classified.

    Input: pandas dataframe with labels in 'fraud' column

    Return: X_train, X_test, y_train, y_test as numpy arrays
    '''

    y = df_fraud.pop('fraud').values
    # Use df_fraud.values once the data is cleaned and all numeric
    X = df_fraud.values
    # Use just a few numeric, full columns to get the model working at the start
    # X = df_fraud[['user_age', 'user_type', 'channels']].values

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    return X_train, X_test, y_train, y_test

def gridsearch(X_train, y_train, params, model):
    '''
    Look for hyperparameters in a SVC model

    Input:
        (1), (2) training data already broken up into X_train and y_train numpy arrays
        (3) a grid of paramters to gridsearch through to make a model
        (4) the sklearn model that will be used in the gridsearch

    Return: The best model that the gridsearch came up with. This model will be fit to the data.
    '''

    #make a model
    mdl = model

    do_search = lambda x: GridSearchCV(estimator=x,
                                       param_grid=params,
                                       cv=3,
                                       scoring='recall',
                                       verbose=3,
                                       return_train_score = True,
                                       n_jobs=-1) \
                                       .fit(X_train, y_train)

    # Gridsearch a SVC model based on the parameter list
    grid_model = do_search(mdl)

    # Get the fit SVC model with the best parameters from the gridsearch
    model_fit = grid_model.best_estimator_
    return model_fit, grid_model

def validation_score(model, X_test, y_test):
    '''
    Uses the model that was created in the gridsearch function to predict categories for the X_test data and see how well the model did.
    f1 score being used to evaluate the preformance of the model

    Input:
        (1) model that preformed the best in the gridsearch
        (2) X data that the model hasn't seen at all yet to validate the model
        (3) True y labels that will be used to score the model

    Return: f1, precision and recall scores based on the previously unseen validataion data
    '''

    y_pred = model.predict(X_test)

    valid_f1 = f1_score(y_test, y_pred)
    valid_prec = precision_score(y_test, y_pred)
    valid_rec = recall_score(y_test, y_pred)

    print '   validation set precision score: {0:.2f}.'.format(valid_prec)
    # print '   this is the ability of the classifier not to label a legal sample as fraudulent.'
    print '   {0:.2f}% of the fraud that our model pointed out was actually fraud.'.format(100*valid_prec)
    print ''
    print '   validation set recall score: {0:.3f}.'.format(valid_rec)
    # print '   this is the ability of the classifier not to label a fraudulent sample as legal.'
    print '   {0:.2f}% of the fraud coming in was actually called out by out model.'.format(100*valid_rec)
    print ''
    print '   validation set f1 score: {0:.3f}.'.format(valid_f1)
    print '   this is a metric that combines the precision and recall scores.'

    return valid_f1, valid_prec, valid_rec

if __name__ == '__main__':
    np.random.seed(60)

    filename = 'clean_df_5.csv'
    df = pd.read_csv(filename)
    # df = df.drop(['old_index','new_index'],axis = 1)
    df_fraud = fraud_column(df)
    # df_fraud.sort_index(axis = 1, inplace = True)
    # svc will be trained and tested through cross validated
    X_train, X_test, y_train, y_test = preprocess(df)

    #dictionary of hyperparameters to test
    #ADD MORE TO SEARCH THROUGH!!
    # params = {'C':np.linspace(.001, 3, 3)}
    '''
    RandomForestClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
    '''
    # note that we have 644 features
    # looping over features to plot affects
    ns = [5, 10, 20, 40, 80 ,160] # max features
    ns = [5, 10, 20, 40, 80, 160] # n estimators could do 320
    ns = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20]# min samples leaf
    ns = [1] # to loop over just one
    trains = []
    tests = []
    for n in ns:
        params = {'n_estimators': [40], #higher
                  'min_samples_leaf': [8], #lower
                  'max_features': [80], #higher
                  'min_samples_split': [3], #higher
                  'oob_score': [False], #True
                  'n_jobs': [3]
                  }

        model = RandomForestClassifier() #AdaBoostClassifier()

        mdl_fit, grid = gridsearch(X_train, y_train, params, model)
        print '----test'
        print ''
        f1, prec, rec = validation_score(mdl_fit, X_test, y_test)
        tests.append([f1,prec,rec])
        print ''
        print '-'*20
        print ''
        print '----train'
        print ''
        f1, prec, rec = validation_score(mdl_fit, X_train, y_train)
        trains.append([f1, prec, rec])

# -------- plotting the variances when changing the
    # plt.style.use('ggplot')
    # f1, prec, rec = zip(*trains)
    # plt.plot(ns,f1,linestyle = '-.',color = 'b',label = 'Train F1')
    # plt.plot(ns,prec,linestyle = '-.',color = 'r',label = 'Train Precision')
    # plt.plot(ns,rec,linestyle = '-.',color = 'g',label = 'Train Recall')
    # f1, prec, rec = zip(*tests)
    # plt.plot(ns,f1,color = 'b',label = 'Test F1')
    # plt.plot(ns,prec,color = 'r',label = 'Test Precision')
    # plt.plot(ns,rec,color = 'g',label = 'Test Recall')
    # plt.xlabel('min_samples_leaf',fontsize = 20)
    # plt.ylabel('Score',fontsize = 20)
    # # plt.legend()
    # # plt.xscale('log')
    # plt.title('6 points, n_estimators = 40, max_features = 80', fontsize = 20)
    # plt.show ()


    # ---------- top features
    top = 30

    feats = mdl_fit.feature_importances_
    i = np.argsort(feats)[::-1]
    cols = np.array(df_fraud.columns.tolist())
    top_n = cols[i][:top]
    top_feats = feats[i][:top]
    tops = zip(top_n,top_feats)
    print tops

    '''

----test

   validation set precision score: 0.96.
   95.66% of the fraud that our model pointed out was actually fraud.

   validation set recall score: 0.853.
   85.31% of the fraud coming in was actually called out by out model.

   validation set f1 score: 0.902.
   this is a metric that combines the precision and recall scores.

--------------------

----train

   validation set precision score: 0.97.
   96.96% of the fraud that our model pointed out was actually fraud.

   validation set recall score: 0.917.
   91.71% of the fraud coming in was actually called out by out model.

   validation set f1 score: 0.943.
   this is a metric that combines the precision and recall scores.

# -----------------------

   top features

 [('quantity_sold', 0.30129828800520336),
 ('sale_duration2', 0.1070573949089052),
 ('pay_other', 0.084884062414186098),
 ('sale_duration', 0.052883591953294541),
 ('user_age', 0.050665812219055306),
 ('gts', 0.047329122753276734),
 ('num_payouts', 0.035383200134365946),
 ('user_type', 0.023739330213435352),
 ('cost', 0.01693296238279306),
 ('has_venue_longitude', 0.010934629470331943),
 ('delivery_method', 0.0087529873849410151),
 ('has_org_name', 0.0079926506335234234),
 ('body_length', 0.007656685738912203),
 ('has_event_published', 0.0057132136445213555),
 ('nlp_293', 0.0050127166065724611),
 ('pay_check', 0.0046048814461881219),
 ('name_length', 0.0044913919930811936),
 ('quantity_total', 0.0038346638574849627),
 ('has_venue_latitude', 0.0037130341905840269),
 ('nlp_178', 0.0034906396282950746),
 ('has_description', 0.0033163817209423941),
 ('lat', 0.0032098326895639772),
 ('nlp_44', 0.0031401549420813891),
 ('has_payee_name', 0.0029241624672553769),
 ('nlp_122', 0.0028707325124463145),
 ('nlp_161', 0.0028513609639768183),
 ('nlp_117', 0.0026447287284730109),
 ('has_venue_address', 0.0025187624785950329),
 ('nlp_52', 0.0025090011220533903),
 ('has_venue_name', 0.0025082686738964978)]
    '''
    # pickle.dump( mdl_fit, open( "random_forest.pkl", "wb"))
