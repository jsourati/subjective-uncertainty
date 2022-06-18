import os, sys
import pdb
import nltk
import codecs
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_parsed_categories(parsed_df, category="RESULT"):
    """Post-processing the result of parsing the abstracts through Prabhakaran's method;
    collecting out the extracted sentences that belong to the "results" into a dictionary
    with keys as the distinct PMIDs and values as the extracted findings of them
    """

    cats = ['BACKGROUND','OBJECTIVE', 'METHOD', 'RESULT', 'CONCLUSION']

    assert category in cats+['ALL'], "Input is not among the available categories."
    
    results = {**{'pmid': []}, **{x: [] for x in cats}}
    tqdm_list = tqdm(range(len(parsed_df)), position=0, leave=True)
    for i in tqdm_list:
        row = parsed_df.iloc[i]
        rtype = row[0]
        
        if rtype == "ABSTRACT":
            if i>0:
                # insert the results into the main dictionary
                results['pmid'] += [pmid]
                for cat in cats:
                    results[cat] += [entry_dict[cat].strip()]
            pmid = row[1]
            entry_dict = {x:'' for x in cats}
        elif rtype not in cats:
            # continue if the row-type is not among the possible ones
            # For instance, there might be 'O'
            continue
            
        else:
            # if category is 'ALL' consider all the statements, and
            # enter them into the place in the result dictionary
            if category == "ALL":
                entry_dict[rtype] += ' '+row[1]
            elif rtype == category:
                entry_dict[rtype] += ' '+row[1]

    if category != 'ALL':
        for cat in set(cats)-{category}:
            del results[cat]

    return results


def measure_uncertainty_df_abstracts(df, sub_unc, block_size, text_column="abstract", save_path=None):
    """Measuring subjective uncertainty in the abstracts/findings saved within a dataframe

    The input dataframe should have at least columns 'abstract' and 'pmid'. The
    input `block_size` determines the number of rows to be considered altogether
    as the input batch when running the uncertainty model.
    """

    lb = 0
    pmids_lst = []
    unc_lst = []
    tqdm_list = tqdm(range(len(df)), position=0, leave=True)

    agg = []
    for lb in range(0, df.shape[0], block_size):
        block = df.iloc[lb : lb + block_size].copy()
        block = block[
            (block[text_column].apply(lambda x: (x != "")))
            & (block[text_column].notnull())
        ].copy()
        abst = block[text_column]
        pids = block["paperid"]

        sents = [[(j, y) for j, y in enumerate(nltk.sent_tokenize(x))] for x in abst]
        flat_list = [
            (pid, j, y) for pid, sublist in zip(pids, sents) for j, y in sublist
        ]
        enu_flat_list = pd.DataFrame(
            flat_list,
            columns=["paperid", "iphrase", "sentence"],
        )
        unc = sub_unc.estimator(enu_flat_list["sentence"].to_list())
        dfa = pd.concat([enu_flat_list, unc], axis=1)

        agg += [dfa]
        tqdm_list.update(block.shape[0])

        if save_path is not None:
            header = True if lb==0 else False
            dfa.to_csv(save_path, mode='a', sep='\t', header=header)

    dft = pd.concat(agg)
    return dft


def eval_zero_shot_biocertainty(sub_unc):

    TRAIN_FILE = "Complete_statements_training_set__ML_model.csv"

    stopwords = nltk.corpus.stopwords.words("english")

    texts = []  # list of text samples
    labels_index = {}  # dictionary mapping label name to numeric id
    labels = []  # list of label ids
    fin = codecs.open(TRAIN_FILE, "r", encoding="utf8")
    for line in fin:
        sent, certain = line.strip().split("\t")
        # sent = [x for x in nltk.word_tokenize(sent) if x not in stopwords]
        # texts.append(' '.join(sent))
        texts.append(sent)
        labels.append(certain)

    P = sub_unc.estimator(texts)

    intervals = np.linspace(0, 1, len(np.unique(labels)))
    inferred_labels = [
        np.min([i for i in range(len(intervals)) if p < intervals[i]]) for p in P
    ]

    return texts, labels, inferred_labels
