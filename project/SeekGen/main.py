from predict import SequencePredict
from train import SequenceTrain

train = SequenceTrain()
seq = SequencePredict()

#train.train(save=True)
train.run_preprocess()
seq.load_models()
seq.run("are you coming")
