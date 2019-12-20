from SeekGen.preprocessor import Preprocess
import pickle
import os
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Flatten, Activation, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Bidirectional, LSTM, LSTMCell
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import model_from_json, Sequential
from sklearn.svm import SVC
from tensorflow.python.keras.metrics import categorical_accuracy

save_path = os.getcwd() + '/SeekGen/savefiles/'

class SequenceTrain:

    def bidirectional_lstm_model(self, seq_shape, vector_len, vocab):
        print('Build LSTM model.')
        inputs = keras.Input(shape=(seq_shape, vector_len), name='digits')
        x = keras.Sequential()(inputs)
        x = Bidirectional(LSTM(512,activation="relu"))(x)
        outputs = Dense(len(vocab),activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='bdrnn')
        return model

    def createTensors(self):
        Y = [self.vocab.index(self.words[i]) for i in range(self.ngram_size, len(self.words))]
        Y = [to_categorical(y,len(self.vocab)) for y in Y]
        X = [np.array([self.embeddings[word] for word in words[i-ngram_size:i]]) for i in range(ngram_size, len(words))]
        X, Y = shuffle(X, Y, random_state=42)
        X_encoded = [encoded[i-self.ngram_size:i] for i in range(self.ngram_size,len(encoded))]
        y_encoded = [encoded[i] for i in range(self.ngram_size,len(encoded))]
        return np.array(X),np.array(Y),np.array(X_encoded), np.array(y_encoded)

    def train(self, save=False):

        processor = self.run_preprocess()

        X, y, X_encoded, y_encoded = self.createTensors()
        tag_model = SVC(gamma='auto',probability=True, class_weight='balanced',verbose=2)
        tag_model.fit(X_encoded, y_encoded)

        X, y = processor.processWords()
        sequence_model =  MLPClassifier(n_iter_no_change =2, verbose=1, activation='tanh', learning_rate='constant', alpha=1e-4, random_state=1, warm_start='full')

        X, y = processor.processWords(flatten=False)
        rnn_model = self.bidirectional_lstm_model(X.shape[1],X.shape[2],processor.vocab)
        rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
        history_rnn = rnn_model.fit(X, y, batch_size=128, shuffle=True, epochs=3, validation_split=0.1)
        print("Training complete!")
        if save==True:
            self.save_models(rnn_model, sequence_model, tag_model)

    def update_models(self):
        #Only update sequence_model and rnn_model
        return

    def save_models(self, rnn_model, sequence_model, tag_model):
        rnn_model.save(save_path+'rnn_model.h5')

        filename = save_path+'wordMLPpredictor.sav'
        pickle.dump(sequence_model, open(filename, 'wb'))

        filename = save_path+'postagger.sav'
        pickle.dump(tag_model, open(filename, 'wb'))
        print("Saved all models!")
