import sys
from sqlalchemy import create_engine
import sqlite3 
# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])
import re
import pandas as pd
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.externals import joblib 

def load_data(database_filepath):
    '''
    load data and parcel X,Y and category names 
    parameters:
    database_filepath: database file path used in process_data.py
    returns:
    X:messages
    Y:categories that messages belong to
    column_names: 36 category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM data", engine)
    X = df.message.values
    cat = df.drop(columns = ['message','original','genre','id'])
    Y = cat.values
    category_names = cat.columns
    return X,Y,category_names 


def tokenize(text):
    '''
    tokenize text 
    parameters:
    text: text that needs to be tokenized
    returns:
    clean_tokens: tokenized text
    '''    
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    build machine learning model and use gridsearch to search for best parameters
    '''   
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    parameters = {
            'vect__ngram_range': ((1, 1), (1, 2)), # only this feature is searched due to time cost and workspace limit
            #'vect__max_df': (0.5, 0.75, 1.0),
            #'vect__max_features': (None, 5000, 10000),
            #'tfidf__use_idf': (True, False),
        }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    evaluate model on testing data and print results 
    parameters:
    model: built model from build_model()
    X_test: test X data
    Y_test: test Y data
    category_names: 36 category names 
    '''
    y_pred = model.predict(X_test)
    for i in range(y_pred.shape[1]):
        print(category_names[i]) 
        print(classification_report(Y_test[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    '''
    save model
    parameters:
    model: built model from build_model()
    model_filepath: name and path to save the model
    '''
    joblib.dump(model, model_filepath) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
