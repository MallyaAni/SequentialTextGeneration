import sqlite3
import pandas as pd
import re
from sklearn.utils import shuffle
import numpy as np
import string
import os
import enchant
import contractions
import nltk
from spacy import displacy
import en_core_web_sm
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils import to_categorical
from nltk.lm import Vocabulary
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

nlp = en_core_web_sm.load()

save_path = os.getcwd() + '/SeekGen/savefiles/'
embed_path = os.getcwd() + '/SeekGen/embeddings/'
glove_input_file = embed_path + 'glove.42B.300d.txt'
word2vec_output_file = embed_path + 'glove.42B.300d.txt.word2vec'

class Preprocess:
    def __init__(self):
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
                    if '\'' in w or '’' in w or '\'' in w:
                        if contractions.fix(w) != w:
                            fixedw = contractions.fix(w).lower()
                            words.append(fixedw)
                    else:
                        words.append(w)
            else:
                if w in['i','a'] or w in string.punctuation:
                    words.append(w)
        return ' '.join(words)

    def message_scrape(self, path, ids):
        conn = sqlite3.connect(path)
        c = conn.cursor()
        output = ""

        for n in range(ids):
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
            output +=' '+' '.join(corpus)

        corpus = re.sub('[^a-zA-Z\'\’\?]', ' ', output)
        corpus = re.sub(r'\s+', ' ', corpus)
        corpus = corpus.lower().split(" ")
        filtered_corpus = self.filter_words(corpus, enchant.Dict("en_US"))
        return filtered_corpus

    def processWords(self):
        self.corpus = self.message_scrape(path=save_path+'chat1.db',ids=100)+' '+ self.message_scrape(path=save_path+'chat2.db',ids=100)
        print("preprocessing words complete!")
        words = nltk.word_tokenize(self.corpus)
        #automate this later
        self.words = list(filter(lambda a: a != 'bet', words))
        self.vocab = list(Vocabulary(self.words))
        self.embeddings = self.runEmbeddings()

    def processTags(self):
        doc = nlp(self.corpus)
        self.tag_encoder = LabelEncoder()
        xtemp = [token.tag_ for token in doc]
        self.tag_encoder.fit(xtemp)
        print("preprocessing tags complete!")

    def runEmbeddings(self, train=False):
        if train:
            glove2word2vec(glove_input_file, word2vec_output_file)
        model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
        print('Word embeddings loaded!')
        return model
