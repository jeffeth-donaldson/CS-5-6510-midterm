import pandas as pd
import numpy as np
import neattext.functions as nfx
import sys
import os

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def main():
    
    print(os.getcwd())
    train_df = pd.read_csv('.\\Problem6\\emotion-dataset.csv', sep=',')
    train_df['Clean Text'] = train_df['Text'].apply(nfx.remove_userhandles)
    train_df['Clean Text'] = train_df['Clean Text'].apply(nfx.remove_stopwords)
    x_train = train_df['Clean Text']
    y_train = train_df['Emotion']

    test_df = pd.read_csv('.\\Problem6\\kaggle_nlp_set\\train.csv', sep=';')
    test_df['Clean Text'] = test_df['Text'].apply(nfx.remove_userhandles)
    test_df['Clean Text'] = test_df['Clean Text'].apply(nfx.remove_stopwords)
    x_test = test_df['Clean Text']
    y_test = test_df['Emotion']

    pipe_ln = Pipeline(steps=[('cv',CountVectorizer()), ('lr',LogisticRegression())])
    pipe_ln.fit(x_train, y_train)

    print(f"score: {pipe_ln.score(x_test, y_test)}")

    sample = "I have a bad case of diahreea"
    print(pipe_ln.predict([sample]))

if __name__ == '__main__':
    main()