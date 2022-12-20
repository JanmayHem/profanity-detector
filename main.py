import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import load_model
import keras.backend as K
import re
import string
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

List = pd.read_csv('Data.csv')
texts = []
for i in List.itertuples():
    texts.append(i[1])
tokenizer = Tokenizer(num_words=250000)
texts = [str(x) for x in texts]
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
data = pad_sequences(sequences, maxlen=1000)


def prediction_fn(text):
    text = text.lower()
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    sample = [text]
    sample = tokenizer.texts_to_sequences(sample)
    sample = pad_sequences(sample, maxlen=1000)
    prediction_score = prediction_model.predict(sample)
    print(prediction_score[0][0])
    if prediction_score[0][0] > 0.65:
        return True
    else:
        return False


def my_custom_activation(x):
    return K.relu(x, alpha=0.1)


prediction_model = load_model('model.h5', custom_objects={'my_custom_activation': my_custom_activation})
