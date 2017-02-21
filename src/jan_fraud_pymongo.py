from pymongo import MongoClient
import pandas as pd
import pickle


def connect_to_mongo(database_name,collection_name):
    # Initial client to connect to mongo
    client = MongoClient()

    # Access/Initiate Database
    database = client[database_name]

    # Access/Initiate Table
    collection = database[collection_name]
    return collection

def write_new_point(database_name,collection_name,point):
    collection = connect_to_mongo(database_name,collection_name)
    collection.insert_one(point)

def make_csv_file(database_name,collection_name,folder, file_name, querry=None):
    collection = connect_to_mongo(database_name,collection_name)

    if querry == None:
        querry = collection.find()

    # to put mongo into a dataframe and pickle the file
    df =  pd.DataFrame(list(querry))
    # folder = where to save file_name = file to save
    df.to_csv('{}/{}.csv'.format(folder,file_name))
    # pickle.dump( df, open( "{}}/{}.pkl".format(folder, file_name), "wb" ) )



if __name__ == '__main__':

# notes on using mongo
    '''
1   // Start the server
2   mongod
3
4   // Start a Mongo shell
5   mongo
6
7   // Show the existing databases
8   show dbs
9
10  // Create a new database called class_db
11  use class_db
12
13  // View all the existing collections (tables)
14  db.getCollectionNames()
16  // Create a collection (table) and insert records into them
17  db.teachers.insert({name: 'E-Rich', age: 25, facial_hair: 'clean'})
18  db.teachers.insert({name: 'Frank', age: 21, friends: ['Adam', 'Cully']})
19  db.teachers.insert({name: 'Neil', age: 55, friends: ['Barack Obama', 'Kanye']})
20
21  // We can then view the first document in our table like so...
22  db.teachers.findOne()
23  db.teachers.findOne().pretty()
    '''
    # to load col_names if needed
    # list(pickle.load(open('col_names.pkl','rb')))

# ------------ pymongo
    # having one point with predicted label come in
    # keys need to be string in pymongo
    new_point = {'name': 'Taylor', 'age': 22, 'friends': ['Lance', 'Gretchen'],'has_kids': 1}


    database_name = 'fraud_test'
    collection_name = 'data'

    # Querry the collection
    # Note this gives back a generator, which you can get results back from one at a time using .next(), or all at once using all_res = list(res)
    # querry = collection.find()

    # adding a new datapoint
    # will add another if run again
    write_new_point(database_name,collection_name,new_point) # function


    # to put mongo into a dataframe and pickle the file
    folder = 'data'
    file_name = 'fraud_test_data'

    make_pkl_file(database_name,collection_name, folder, file_name, querry=None) #fuction
