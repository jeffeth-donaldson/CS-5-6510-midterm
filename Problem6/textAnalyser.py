import pandas as pd
import numpy as np
import neattext.functions as nfx
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

def main():
    text_df = pd.read_csv(sys.argv[1])
    text_df['Clean Text'] = text_df['Text'].apply(nfx.remove_userhandles)
    text_df['Clean Text'] = text_df['Text'].apply(nfx.remove_stopwords)

    xfeatures = text_df['Clean Text']
    ylabels = text_df['Emotion']

    x_train,x_test,y_train,y_test = train_test_split(xfeatures, ylabels, test_size=0.3,random_state=7)

    pipe_ln = Pipeline(steps=[('cv',CountVectorizer()), ('lr',LogisticRegression())])
    pipe_ln.fit(x_test, y_train)

    print(f"score: {pipe_ln.score(x_test, y_test)}")

if __name__ == '__main__':
    main()