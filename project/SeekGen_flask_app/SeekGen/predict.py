import pickle
import spacy
import getch
import nltk
import re
import os
import numpy as np
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from SeekGen.train import SequenceTrain

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import model_from_json, Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Flatten, Activation, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Bidirectional, LSTM, LSTMCell
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.metrics import categorical_accuracy
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras import backend as K

save_path = os.getcwd() + '/SeekGen/savefiles/'

class SequencePredict:

    def __init__(self, processor):
        self.processor = processor
        self.ngram_size = 3

    def load_models(self):
        self.rnn_model = tf.keras.models.load_model(save_path+'rnn_model.h5')
        self.tag_model = pickle.load(open(save_path+'postagger.sav', 'rb'))
        print("loaded models!")

    def input_tag_to_prediction(self, inp):
        doc = nlp(inp.lower())
        tags = [token.tag_ for token in doc]
        X_encoded = self.processor.tag_encoder.transform(tags)
        y_pred = self.tag_model.predict_proba(X_encoded.reshape(1, -1))[0]
        indices = (y_pred).argsort()[::-1][:3]
        return [self.processor.tag_encoder.inverse_transform([idx])[0] for idx in indices]

    def input_word_to_prediction(self, sent):
        words = nltk.word_tokenize(sent.lower())
        x = np.array([self.processor.embeddings[word] for word in words])
        words_tf = x.reshape(1,x.shape[0],x.shape[1])
        y_pred = np.array(self.rnn_model.predict(words_tf)[0])
        indices = (y_pred).argsort()[::-1][:3]
        return [self.processor.vocab[idx] for idx in indices]

    def prediction_union(self, sent):
        tags = self.input_tag_to_prediction(sent)
        superset_tags = [tag[:2] for tag in tags]
        predictions = self.input_word_to_prediction(sent)
        prediction_tags = [nlp(word)[0].tag_[:2] for word in predictions]
        filtered_predictions = [word for word in predictions if nlp(word)[0].tag_[:2] in superset_tags]
        return predictions, filtered_predictions

    def run(self, sent):
        predictions, filtered_predictions  = self.prediction_union(sent)
        return predictions
        # if sent is None:
        #     sent = input()
        # else:
        #     sent += " " + input()
        # print(sent)
        # if len(sent.split(" ")) < self.ngram_size:
        #     self.run(sent)
        #
        # trigram_sent = " ".join(nltk.word_tokenize(sent)[-3:])
        # combined_predictions, filtered_predictions  = self.prediction_union(trigram_sent)
        # print(filtered_predictions)
        # cur_word = ""
        # while True:
        #     char = getch.getch()
        #     print(char)
        #     cur_word+=char
        #
        #     if char=='\n':
        #         sent = " ".join(nltk.word_tokenize(sent)+[cur_word])
        #         self.run(sent)
        #
        #     if char.isdigit():
        #         cur_word = filtered_predictions[int(char)-1]
        #         sent = " ".join(nltk.word_tokenize(sent)+[cur_word])
        #         self.run(sent)
        #
        #     if len(list(filter(lambda x: cur_word in x, filtered_predictions)))==0 and len(cur_word)==2 and len(list(filter(lambda x: cur_word in x, combined_predictions)))!=0:
        #         choices = [possible_choice for possible_choice in combined_predictions if cur_word in possible_choice]
        #         print(choices, "press tab to autocomplete")
        #
        #     if char=="\t":
        #         sent = " ".join(nltk.word_tokenize(sent)+[choices[0]])
        #         self.run(sent)
