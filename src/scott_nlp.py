import pandas as pd
import spacy
import string
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk.stem.snowball import EnglishStemmer
import re
import cPickle as pickle
from sklearn.decomposition import PCA
import numpy as np

def load_data():
    '''
    Loads the data and inserts three columns:
    'fraud',
    'description_clean_html', the description with html stripped, useful for presenting single sentences.
    'description_clean_nlp', the description with words lowercased, and no punctuation or stopwords, useful for tfidf.
    '''
    df = pd.read_json('data/data.json')
    df['fraud'] = \
            (df['acct_type']=='fraudster_event') | \
            (df['acct_type']=='fraudster') | \
            (df['acct_type']=='fraudster_att')
    df['description_clean_html'] = df['description'].map(clean_html)
    df['description_clean_nlp'] = df['description_clean_html'].map(clean_document)
    return df

def save_csv():
    '''
    Saves the original data to a csv.
    Needed to overcome memory errors (can load later using a generator)
    '''
    df = load_data()
    df.to_csv('data/data.csv')

def clean_html(raw_html):
    '''
    Removes all html tags from a string.
    Taken from here: http://stackoverflow.com/questions/9662346/python-code-to-remove-html-tags-from-a-string
    '''
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def clean_document(document):
    '''
    Cleans a document for further processing by NLP algorithms.
    Converts all word to lowercase, removes punctuation, and removes stop words.
    '''
    document = document.split()
    document = [word.lower() for word in document]
    document = [filter(lambda c: c in string.letters, word) for word in document]
    document = filter(lambda w: w not in ENGLISH_STOP_WORDS,document)
    es = EnglishStemmer()
    document = [es.stem(word) for word in document]
    document = filter(lambda w: w != '',document)
    document = string.join(document, ' ')
    return document

def sample_fraud_descriptions(df, n=20, column='description_clean_html'):
    '''
    Returns sample descriptions of fraud and valid.
    '''
    fraud_descriptions = df[df['fraud']==True][column].sample(n=n, random_state=100)
    valid_descriptions = df[df['fraud']==False][column].sample(n=n, random_state=100)
    return fraud_descriptions, valid_descriptions

def save_nlp_vectors(column='description'):
    nlp = spacy.load('en')
    description_vectors = compute_description_vectors(nlp)
    with open('data/description_vectors.pkl', 'w') as f:
        pickle.dump(description_vectors, f)

def compute_nlp_vectors(nlp, column='description'):
    gen = pd.read_csv('data/data.csv', chunksize=1000, encoding='utf-8')
    description_vectors = []
    batch = 1
    for df in gen:
        print 'batch {}'.format(batch)
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
        batch += 1
    description_vectors = np.array(description_vectors)
    return description_vectors

def compute_nlp_vectors_from_df(nlp, df, column='description'):
    gen = pd.read_csv('data/data.csv', chunksize=1000, encoding='utf-8')
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

def save_pca(n_components=10):
    with open('data/description_vectors.pkl') as f:
        description_vectors = pickle.load(f)
    model = compute_pca(description_vectors)
    with open('data/description_vectors_pca_model.pkl', 'w') as f:
        pickle.dump(model, f)
    with open('data/description_vectors_pca_components.pkl', 'w') as f:
        pickle.dump(model.components_, f)

def compute_pca(description_vectors, n_components=10):
    pca = PCA(n_components=n_components)
    pca.fit(description_vectors)
    return pca

if __name__=='__main__':
    nlp = spacy.load('en')
    df = load_data()
