from preprocessor import Preprocess
import pickle
from sklearn.neural_network import MLPClassifier
from tensorflow import keras
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, Flatten, Activation, MaxPooling2D, Conv2D, GlobalAveragePooling2D, Bidirectional, LSTM, LSTMCell
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.models import model_from_json, Sequential
from sklearn.svm import SVC
from tensorflow.python.keras.metrics import categorical_accuracy

class SequenceTrain:

    def bidirectional_lstm_model(self, seq_shape, vector_len, vocab):
        print('Build LSTM model.')
        inputs = keras.Input(shape=(seq_shape, vector_len), name='digits')
        x = keras.Sequential()(inputs)
        x = Bidirectional(LSTM(512,activation="relu"))(x)
        outputs = Dense(len(vocab),activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs, name='bdrnn')
        return model

    def run_preprocess(self):
        processor = Preprocess(2)
        processor.processWords()
        processor.processTags()
        return processor

    def train(self, save=False):

        processor = self.run_preprocess()

        X_encoded, y_encoded = processor.processTags()
        tag_model = SVC(gamma='auto',probability=True, class_weight='balanced',verbose=2, max_iter=10)
        tag_model.fit(X_encoded, y_encoded)

        X, y = processor.processWords()
        sequence_model =  MLPClassifier(n_iter_no_change =2, max_iter = 10, verbose=1, activation='tanh', learning_rate='constant', alpha=1e-4, random_state=1, warm_start='full') #picked 60 as limit because it overfits after
        sequence_model.fit(X,y)

        X, y = processor.processWords(flatten=False)
        rnn_model = self.bidirectional_lstm_model(X.shape[1],X.shape[2],processor.vocab)
        rnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
        history_rnn = rnn_model.fit(X, y, batch_size=32, shuffle=True, epochs=2, validation_split=0.1)
        print("Training complete!")
        if save==True:
            self.save_models(rnn_model, sequence_model, tag_model)

    def save_models(self, rnn_model, sequence_model, tag_model):
        rnn_model.save('/Users/animallya/Desktop/SequentialTextGeneration/project/savefiles/rnn_model.h5')

        filename = '/Users/animallya/Desktop/SequentialTextGeneration/project/savefiles/wordMLPpredictor.sav'
        pickle.dump(sequence_model, open(filename, 'wb'))

        filename = '/Users/animallya/Desktop/SequentialTextGeneration/project/savefiles/postagger.sav'
        pickle.dump(tag_model, open(filename, 'wb'))
        print("Saved all models!")
