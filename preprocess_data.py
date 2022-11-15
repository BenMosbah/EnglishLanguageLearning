import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy
raw_data = pd.read_csv('Data/train.csv')
preprocessed_data = raw_data.copy()
preprocessed_data.loc[:, 'full_text'] = preprocessed_data['full_text'].str.strip()
preprocessed_data.loc[:, 'NbWords'] = preprocessed_data['full_text'].str.split(' ').str.len()
preprocessed_data.loc[:, 'NbCharacters'] = preprocessed_data['full_text'].str.len()
preprocessed_data.loc[:, 'NbSentences'] = preprocessed_data['full_text'].str.count("[.]")
punctuation_regex = '[,]|[;]|[?]|[!]'
preprocessed_data.loc[:, 'NbPunctutation'] = preprocessed_data['full_text'].str.count(punctuation_regex)
preprocessed_data.loc[:, 'NbQuestions'] = preprocessed_data['full_text'].str.count('[?]')
preprocessed_data.loc[:, 'NbExclamations'] = preprocessed_data['full_text'].str.count('[!]')

all_corpus = list(preprocessed_data['full_text'])
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(all_corpus)
sequences = tokenizer.texts_to_sequences(all_corpus)
padded = pad_sequences(sequences)
#padded is ready to be fed to a neural network. I think
