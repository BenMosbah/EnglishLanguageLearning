import pandas as pd
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
