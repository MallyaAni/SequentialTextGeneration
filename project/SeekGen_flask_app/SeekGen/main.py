from predict import SequencePredict
from preprocessor import Preprocess

#train.train(save=True)
processor = Preprocess()
processor.processWords()
processor.processTags()

seq = SequencePredict(processor)

seq.load_models()
seq.run()
