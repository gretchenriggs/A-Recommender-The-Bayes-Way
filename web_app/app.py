from flask import Flask, render_template, url_for, redirect, Markup
from flask_pymongo import PyMongo
import os
import cPickle as pickle
import requests
from jan_fraud_pymongo import write_new_point, make_csv_file
from web_app_preprocessor import Preprocessor

app = Flask(__name__)
app.jinja_env.autoescape = False
mongo = PyMongo(app)
preprocessor = Preprocessor()
with open("../random_forest.pkl") as f_un:
    model = pickle.load(f_un)

# naming our mongo database
database_name = 'fraud_test2'
collection_name = 'data'

# from models import Result
# home page
@app.route('/', methods = ['GET', 'POST'])
def home_page():
    online_users = mongo.db.users.find({'online': True})
    return render_template('index.html', online_users=online_users)

@app.route('/predict', methods=['GET'])
def get_and_score():
    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    X_json = r.json()



    X_input = preprocessor.transform(X_json)
    X_input = X_input.fillna(0)
    # for col in sorted(X_input):
        # print X_input[col]
    y = model.predict(X_input)
    description = X_json['description']
    # X_json['prediction'] = y
    # return X_json

    # ----- write to the mongo surver
    X_json['y'] = y[0]

    write_new_point(database_name,collection_name,X_json) # function
    print_dict ={0:'Not fraud',1:'Potential FRAUD'}
    return render_template('predict.html', description=Markup(description), prediction=print_dict[y[0]])

# Score will show the static scores from the Classification models that were
#   tested.
@app.route('/score', methods = ['GET', 'POST'])
def score():
    return render_template('score.html')

# Scope will detail what is being done with the Fraud Detection project
@app.route('/scope')
def scope():
    return render_template('scope.html')

# Dashboard will show live analysis of new data points
@app.route('/dashboard')
def dashboard():
    return 'dashboard'


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8106, debug=True)
