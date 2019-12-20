				SEEKGEN v2.0

Run python3 app.py after cd-ing into SeekGen_flask_app folder. Seekgen_flask_app folder contains all the python scripts that are used to run the web app.

predictive_text.ipynb is the Jupyter notebook that contains our final code.
SeekGen folder contains 4 python files that are run as scripts during the flask application web app call.

Eval folder contains the csv file that was used as testing data manually labeled.

Glove embeddings arent pushed onto GitHub because its 5gb in size.


Pos tagger was trained on NLTK web-text data, since the language is more structured than on iMessage casual conversations. This helps us eliminate false predictions by learning good language transitions.

Bidirectional LSTMs and Multilayer perceptron models provide a combination of contextual and sequential predictions and were trained on scraped iMessage data from Ani's iPhone 11 Pro.

Trained models are stored on GCP with appropriate security limitations.
