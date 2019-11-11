import sqlite3
import pandas as pd
import re
import enchant
from sklearn.utils import shuffle
import numpy as np
import string
import contractions
import nltk
from spacy import displacy
import en_core_web_sm
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.python.keras.utils import to_categorical

nlp = en_core_web_sm.load()

class Preprocess:
    def __init__(self,id):
        self.id = id
        self.vectorizer = None
        self.vocab = None
        self.words = None
        self.tag_encoder = None
        self.ngram_size = 3

    def filter_words(self, text, global_vocab):
        words = []
        for w in text:
            w = w.lower()
            if len(w)>1:
                if global_vocab.check(w)==True:
                    words.append(w)
                elif '\'' in w or '’' in w:
                    if contractions.fix(w) != w:
                        fixedw = contractions.fix(w).lower()
                        words.append(fixedw)
            else:
                if w in['i','a'] or w in string.punctuation:
                    words.append(w)
        return ' '.join(words)

    def message_scrape(self, path, n):
        conn = sqlite3.connect(path)
        c = conn.cursor()
        cmd1 = 'SELECT ROWID, text, handle_id, \
                    datetime(date + strftime(\'%s\',\'2001-01-01\'), \'unixepoch\') as date_utc \
                    FROM message T1 \
                    INNER JOIN chat_message_join T2 \
                        ON T2.chat_id='+str(n)+ ' \
                        AND T1.ROWID=T2.message_id \
                    ORDER BY T1.date'
        c.execute(cmd1)
        df_msg = pd.DataFrame(c.fetchall(), columns=['id', 'text', 'sender', 'time'])
        corpus = df_msg.text.tolist()
        corpus = ' '.join(corpus)
        corpus = re.sub('[^a-zA-Z\'\’\?]', ' ', corpus)
        corpus = re.sub(r'\s+', ' ', corpus)
        corpus = corpus.lower().split(" ")
        filtered_corpus = self.filter_words(corpus, enchant.Dict("en_US"))
        return filtered_corpus

    def processWords(self, flatten=True):

        corpus = self.message_scrape(path='/Users/animallya/Desktop/chat.db',n=self.id)
        self.words = nltk.word_tokenize(corpus)
        self.vocab = list(nltk.lm.Vocabulary(self.words))
        self.vectorizer = CountVectorizer(vocabulary=self.vocab)
        self.vectorizer.fit(self.words)

        if self.id==2:
        #automate this later
            self.words = list(filter(lambda a: a != 'bet', self.words))

        Y = [self.vocab.index(self.words[i]) for i in range(self.ngram_size, len(self.words))]
        if flatten == True:
            X = [self.vectorizer.transform(self.words[i-self.ngram_size:i]).toarray().flatten() for i in range(self.ngram_size, len(self.words))]
        else:
            X = [self.vectorizer.transform(self.words[i-self.ngram_size:i]).toarray().astype('float64') for i in range(self.ngram_size, len(self.words))]
        Y = [to_categorical(y,len(self.vocab)) for y in Y]
        X, Y = shuffle(X, Y, random_state=42)

        return np.array(X),np.array(Y)

    def processTags(self):
        doc = nlp(' '.join(self.words))
        self.tag_encoder = LabelEncoder()

        xtemp = [token.tag_ for token in doc]
        encoded = self.tag_encoder.fit_transform(xtemp)
        X_encoded = [encoded[i-self.ngram_size:i] for i in range(self.ngram_size,len(encoded))]
        y_encoded = [encoded[i] for i in range(self.ngram_size,len(encoded))]
        return np.array(X_encoded), np.array(y_encoded)
