from flask import Flask, render_template, request, url_for, session, redirect, escape, make_response, flash
import json
import os
import nltk
import requests
import random
import subprocess
from SeekGen.predict import SequencePredict
from SeekGen.preprocessor import Preprocess

processor = Preprocess()
processor.processWords()
processor.processTags()

seq = SequencePredict(processor)
seq.load_models()


prev = []
sentence = ""
flag = False


app = Flask(__name__)
app.secret_key = b'\xdc\xebBG%jV*\x1e\x9f*l\x98\xbb\x89\x0f'


@app.route('/')
@app.route('/index/')
def hello_world(name=None):
    return render_template('index.html', name=name)

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
        global prev, sentence, seq, flag
        res = []

        if request.method == 'POST':
            if request.form.get('ignore'):
                user_input = request.form['override']
            else:
                user_input = request.form['response']
            
            sentence += " " + user_input
            tok_input = nltk.word_tokenize(user_input)

            #this flag is set when two words are seen so that three words can be taken for a prediction input
            if not flag:
                chunk = nltk.word_tokenize(sentence)
                if len(chunk) == 2:
                    flag = True
                    prev = chunk

            else:
                new_word = prev + tok_input
                res = seq.run(" ".join(new_word))
                prev = new_word[-2:]

            return render_template('prediction.html', sentence=sentence, options=res)

        return render_template('prediction.html')


if __name__ == "__main__":
    app.run(debug=True)
