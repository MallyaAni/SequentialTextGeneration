import pickle
import spacy
import nltk
import numpy as np
from spacy import displacy
import en_core_web_sm
nlp = en_core_web_sm.load()
from train import SequenceTrain

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

save_path = '/Users/animallya/Desktop/SequentialTextGeneration/project/savefiles/'

class SequencePredict:

    def __init__(self):
        trainer = SequenceTrain()
        processor = trainer.run_preprocess()
        self.vectorizer = processor.vectorizer
        self.tag_encoder = processor.tag_encoder
        self.vocab = processor.vocab

    def load_models(self):
        self.rnn_model = tf.keras.models.load_model(save_path+'rnn_model.h5')
        self.sequence_model = pickle.load(open(save_path+'wordMLPpredictor.sav','rb'))
        self.tag_model = pickle.load(open(save_path+'postagger.sav', 'rb'))
        print("loaded models!")

    def input_tag_to_prediction(self, inp):
        doc = nlp(inp.lower())
        tags = [token.tag_ for token in doc]
        X_encoded = self.tag_encoder.transform(tags)
        y_pred = self.tag_model.predict_proba(X_encoded.reshape(1, -1))[0]
        indices = (y_pred).argsort()[::-1][:3]
        return [self.tag_encoder.inverse_transform([idx])[0] for idx in indices]

    def input_word_to_prediction(self, inp):
        words = nltk.word_tokenize(inp.lower())
        words_tf = np.array(self.vectorizer.transform(words).toarray()).flatten().reshape(1, -1)
        y_pred = self.sequence_model.predict_proba(words_tf)[0]
        indices = (y_pred).argsort()[::-1][:3]
        return [self.vocab[idx] for idx in indices]

    def rnn_predict(self, sent):
        words = nltk.word_tokenize(sent.lower())
        x = np.array(self.vectorizer.transform(words).toarray().astype('float64'))
        words_tf = x.reshape(1,x.shape[0],x.shape[1])
        y_pred = np.array(self.rnn_model.predict(words_tf)[0])
        indices = (y_pred).argsort()[::-1][:3]
        return [self.vocab[idx] for idx in indices]

    def prediction_union(self, sent):
        tags = self.input_tag_to_prediction(sent)
        superset_tags = [tag[:2] for tag in tags]
        print(tags, superset_tags)
        combined_predictions = self.rnn_predict(sent)
        combined_predictions.extend(x for x in self.input_word_to_prediction(sent) if x not in combined_predictions)
        prediction_tags = [nlp(word)[0].tag_[:2] for word in combined_predictions]
        print(prediction_tags)
        filtered_predictions = [word for word in combined_predictions if nlp(word)[0].tag_[:2] in superset_tags]

        return combined_predictions, filtered_predictions

    def run(self, sent):
        out = sent
        while True:
            print(out)
            combined_predictions, filtered_predictions  = self.prediction_union(sent)
            #print(combined_predictions)
            print(filtered_predictions)
            inp = input()
            if inp=="exit":
                return out
            out+= " " +inp
            l = len(inp.split())
            sent = " ".join(nltk.word_tokenize(sent)[l:]+[inp])
        print(out)