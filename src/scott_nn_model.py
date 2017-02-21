import pandas as pd
import numpy as np
from taylor_eda import fraud_column
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import Adadelta

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
                                       n_jobs=-1) \
                                       .fit(X_train, y_train)

    # Gridsearch a SVC model based on the parameter list
    grid_model = do_search(mdl)

    # Get the fit SVC model with the best parameters from the gridsearch
    model_fit = grid_model.best_estimator_
    return model_fit

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

class NNModel(BaseEstimator):

    def __init__(self, num_hidden=100, nb_epoch=10):
        self.num_hidden = num_hidden
        self.nb_epoch = nb_epoch
    
    # def fit(self, X_train, y_train, num_hidden=100, nb_epoch=10):
    def fit(self, X_train, y_train):
        self.scaler = StandardScaler()
        self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        y_train_ohe = np_utils.to_categorical(y_train, nb_classes=2)
        model = self.build_model(num_inputs=X_train.shape[1], num_hidden=self.num_hidden)
        model.fit(X_train, y_train_ohe, nb_epoch=self.nb_epoch)
        self.model = model

    def build_model(self, num_inputs, num_hidden):
        layer1 = Dense(
                input_dim=num_inputs,
                output_dim=num_hidden,
                init='uniform',
                activation='tanh'
                )
        layer2 = Dense(
                input_dim=num_hidden,
                output_dim=2,
                init='uniform',
                activation='softmax'
                )
        opt = Adadelta()

        model = Sequential() # sequence of layers
        model.add(layer1)
        model.add(layer2)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def predict(self, X_test):
        X_test = self.scaler.transform(X_test)
        return self.model.predict_classes(X_test)

if __name__ == '__main__':
    filename = 'data/clean_df_4.csv'
    df = pd.read_csv(filename)
    df_fraud = fraud_column(df)

    # svc will be trained and tested through cross validated
    X_train, X_test, y_train, y_test = preprocess(df)

    #dictionary of hyperparameters to test
    #ADD MORE TO SEARCH THROUGH!!
    # params = {'C':np.linspace(.001, 3, 3)}
    # params = {'n_estimators': [50],
    #           'learning_rate': [0.5]}
    params = {'num_hidden': [10, 100, 500], 'nb_epoch': [10, 50]}

    model = NNModel()
    mdl_fit = gridsearch(X_train, y_train, params, model)

    f1, prec, rec = validation_score(mdl_fit, X_test, y_test)
