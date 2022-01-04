import os, sys
import pdb
import nltk
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from transformers import pipeline

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

path = os.path.join(os.getcwd(), 'certainty-estimator')
sys.path.insert(0,path)
from certainty_estimator.predict_certainty import CertaintyEstimator

PATH_TO_LSTM = 'BioCertainty/data/'
STOPWORDS = nltk.corpus.stopwords.words('english')

class SubjectiveUncertainty(object):
    def __init__(self, utype, **kwargs):

        if utype=='zero-shot':
            device = kwargs.get('device', -1)
            self.summarizer = kwargs.get('summarizer', np.max)
            self.keywords = kwargs.get('keywords', ["certain", "uncertain"])

            assert len(self.keywords)==2, 'Number of keywords should be exactly two, {} given'.format(len(self.keywords))
            
            self.model = pipeline("zero-shot-classification", device=device)
            self.estimator = lambda sents: zero_shot_estimator(sents,
                                                               self.model,
                                                               self.keywords)

        elif utype=='lstm':
            self.summarizer = kwargs.get('summarizer', np.max)
            self.tokenizer, self.model, self.max_seq_len = load_lstm()
            self.estimator = lambda sents: lstm_estimator(sents,
                                                          self.tokenizer,
                                                          self.model,
                                                          self.max_seq_len)

        elif utype=='scibert':
            os.chdir('certainty-estimator')
            self.cert_estimator = CertaintyEstimator('sentence-level')
            os.chdir('..')

            device = kwargs.get('device', -1)
            self.summarizer = kwargs.get('summarizer', np.min)
            if device > -1:
                self.cert_estimator.model = self.cert_estimator.model.to('cuda:{}'.format(device))
                self.cert_estimator.cuda = True

            self.model = self.cert_estimator.model
            self.estimator = lambda sents: self.cert_estimator.predict(sents)
            
                    

    
def load_lstm():

    training_set = PATH_TO_LSTM + '/training_set.csv'
    model_json = PATH_TO_LSTM + '/model.json'
    model_h5 = PATH_TO_LSTM + '/model.h5'

    MAX_NB_WORDS = 6660

    # loading the training text
    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    fin = (codecs.open(training_set, "r",  encoding='utf8'))
    maxlen = 0
    for line in fin:
        sent = (line.strip().replace('\n', ' '))
        sent = [x for x in nltk.word_tokenize(sent) if x not in STOPWORDS]
        texts.append(' '.join(sent))
        if len(sent) > maxlen:
            maxlen = len(sent)
    fin.close()


    # building tokenizer 
    max_seq_len = maxlen
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)

    # loading the model
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(model_h5)

    return tokenizer, model, max_seq_len


def lstm_estimator(sents, tokenizer, model, max_seq_len):

    # initial tokenizing the sentences
    sents = [np.array(nltk.word_tokenize(s.strip())) for s in sents]
    sents = [x[~np.isin(x,STOPWORDS)] for x in sents]
    sents = [' '.join(x) for x in sents]

    # converting texts to sequences of integers and padding to a fixed-length vector
    seqs = tokenizer.texts_to_sequences(sents)
    pseqs = pad_sequences(seqs, maxlen=max_seq_len)
    pseqs = tf.convert_to_tensor(pseqs)

    # running the model
    preds = model.predict_on_batch(pseqs)
    #classes = np.argmax(preds,axis=1)

    # returning mean of the predicted classes 
    return preds[:,1] + 2*preds[:,2]
                                                

def zero_shot_estimator(sents, model, keywords):
    """The estimator returns the probability of the second keyword,
    i.e., `keywords[1]`
    """

    res = model(sents,
                candidate_labels=keywords,
                multi_class=False)
    if len(sents)==1:
        res = [res]
    
    scores = np.array([np.array(x['scores'])[np.array(x['labels'])==keywords[1]][0]
                       for x in res])

    return scores

    


        
