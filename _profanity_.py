import numpy as np
import pandas as pd
import re
import warnings
import string
import unidecode
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential, load_model
from keras.layers.core import Embedding, Dense, Dropout
from keras.layers import GRU
import keras.backend as K
from keras.utils import get_custom_objects
import os
warnings.filterwarnings('ignore')

for dir_name, _, filenames in os.walk('Input Files'):
    for filename in filenames:
        print(os.path.join(dir_name, filename))
lis1 = pd.read_csv('Input Files/Comment Classification/train.csv')
lis2 = pd.read_csv('Input Files/Toxicity Severity/validation_data.csv')
lis1['compound'] = lis1.sum(axis=1)


def pos(x):
    if x > 0:
        return 1
    else:
        return 0


lis1['y'] = lis1['compound'].apply(lambda x: pos(x))
lis1 = lis1[['comment_text', 'y']]

less_arr = pd.DataFrame(lis2['less_toxic']).rename(columns={'less_toxic': 'comment_text'}, inplace=False)
more_arr = pd.DataFrame(lis2['more_toxic']).rename(columns={'more_toxic': 'comment_text'}, inplace=False)
more_arr['y'] = 1
less_arr['y'] = 0
lis1 = pd.concat([less_arr, more_arr, lis1])

contractions_dict = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}
contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(text, contractions_dict1=contractions_dict):
    def replace(match):
        return contractions_dict1[match.group(0)]
    return contractions_re.sub(replace, text)


lis1['comment_text'] = lis1['comment_text'].apply(lambda x: expand_contractions(x))
lis1['comment_text'] = lis1['comment_text'].str.lower()
lis1['comment_text'] = lis1['comment_text'].apply(lambda x: re.sub('[%s]' % re.escape(string.punctuation), '', x))
lis1['comment_text'] = lis1['comment_text'].apply(lambda x: re.sub('\w*\d\w*', '', x))
lis1[lis1['y'] >= 0].hist()
lis1 = pd.concat([lis1[lis1['y'] > 0], lis1[lis1['y'] == 0].sample(int(len(lis1[lis1['y'] > 0]) * 1.5))], axis=0).sample(frac=1)
lis1[lis1['y'] >= 0].hist()
lis1['comment_text'] = lis1['comment_text'].apply(lambda x: unidecode.unidecode(x))

texts, outputs = [], []
for i in lis1.itertuples():
    texts.append(i[1])
    outputs.append(i[-1])

tokenizer = Tokenizer(num_words=250000)
t_samples = 100000
v_samples = 15832
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
data = pad_sequences(sequences, maxlen=1000)
outputs = np.asarray(outputs)
indices = np.arange(data.shape[0])
data = data[indices]
labels = outputs[indices]
Xt, Xv = data[:t_samples], data[t_samples: t_samples + v_samples]
yt, yv = outputs[:t_samples], outputs[t_samples: t_samples + v_samples]
f = open(os.path.join('Input Files/VecToWord', 'glove.twitter.27B.100d.txt'), encoding='utf-8')
embeddings_index = {}
for line in f:
    value = line.split()
    word = value[0]
    coefs = np.asarray(value[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 100
embedding_matrix = np.zeros((1193514, embedding_dim))
for word, i in word_index.items():
    if i < 200:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
model = Sequential()
model.add(Embedding(1193514, embedding_dim, input_length=1000))
model.add(GRU(64))


def my_custom_activation(x):
    return K.relu(x, alpha=0.1)


get_custom_objects().update({'my_custom_activation': my_custom_activation})
model.add(Dropout(0.08))
model.add(Dense(1, activation=my_custom_activation))
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = True
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(Xt, yt, epochs=4, batch_size=32, validation_data=(Xv, yv))

model.save("model.h5")
load_model("model.h5")

sample = ['what the fuck', 'whos nigga is this dick', 'f*ck you digger', 'you are a fucking idiot', 'Wikipedia is amazing. I update a few articles and people who have no knowledge about the topic vandalize it and make threats if I correct their vandalizm. Im about to bail out.', 'clean text']
sample = tokenizer.texts_to_sequences(sample)
sample = pad_sequences(sample, maxlen=1000)
predictions = model.predict(sample)
print(predictions)
