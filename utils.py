import os, sys
import pdb
import nltk
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_parsed_findings(parsed_df):
    """Post-processing the result of parsing the abstracts through Prabhakaran's method;
    collecting out the extracted sentences that belong to the "results" into a dictionary 
    with keys as the distinct PMIDs and values as the extracted findings of them
    """
    results = {}
    tqdm_list = tqdm(range(len(parsed_df)), position=0, leave=True)
    for i in tqdm_list:
        row = parsed_df.iloc[i]
        rtype = row[0]
        if rtype=='ABSTRACT':
            # if we are not at the first iteration, the results should be updated
            if i>0:
                results[pmid] = pmid_results
            pmid = row[1]
            pmid_results = []
        else:
            if rtype=='RESULT':
                pmid_results += [row[1]]

        # in the last iteration, update the results for the last PMID
        if i==len(parsed_df)-1:
            results[pmid] = pmid_results

    return results


def measure_uncertainty_df_abstracts(df, sub_unc, block_size, column='abstract'):
    """Measuring subjective uncertainty in the abstracts/findings saved within a dataframe

    The input dataframe should have at least columns 'abstract' and 'pmid'. The
    input `block_size` determines the number of rows to be considered altogether
    as the input batch when running the uncertainty model.
    """

    #df.fillna('',inplace=True)

    lb = 0
    pmids_lst = []
    unc_lst = []
    tqdm_list = tqdm(range(len(df)), position=0, leave=True)
    while lb<len(df):
        abst = [x for x in df.iloc[lb:lb+block_size,][column] if len(x)>0]
        # list of lists
        sents = [nltk.sent_tokenize(x) for x in abst]
        nsents = [len(x) for x in sents]
        # list
        sents = sum(sents, [])

        pmids = [df.iloc[lb+i].pmid
                 for i,x in enumerate(df.iloc[lb:lb+block_size,].abstract) if len(x)>0]

        unc = sub_unc.estimator(sents)

        for i in range(len(nsents)):
            _lb = int(np.sum(nsents[:i]))
            _ub = int(np.sum(nsents[:i+1]))
            summ = sub_unc.summarizer(unc[_lb:_ub])
            #if inplace:
            #    df.loc[df['pmid']==pmids[i],['zero_shot_su']]= avg

            pmids_lst += [pmids[i]]
            unc_lst += [summ]

        lb += block_size
        tqdm_list.update(min(block_size,len(df)-lb+block_size))

    return pmids_lst, unc_lst


def eval_zero_shot_biocertainty(sub_unc):
    
    TRAIN_FILE = 'Complete_statements_training_set__ML_model.csv'

    stopwords = nltk.corpus.stopwords.words('english')
    
    texts = []   # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    fin = codecs.open(TRAIN_FILE, "r", encoding='utf8')
    for line in fin:
        sent, certain = line.strip().split("\t")
        # sent = [x for x in nltk.word_tokenize(sent) if x not in stopwords]
        # texts.append(' '.join(sent))
        texts.append(sent)
        labels.append(certain)

    P = sub_unc.estimator(texts)

    intervals = np.linspace(0,1,len(np.unique(labels)))
    inferred_labels =  [np.min([i for i in range(len(intervals)) if p<intervals[i]])
                        for p in P]

    return texts, labels, inferred_labels
