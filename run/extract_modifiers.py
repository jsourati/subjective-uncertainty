import argparse
import os

import pandas as pd
import spacy

from subj_unc.dep_tree import phrase_to_deptree


conditions = [
    {"tag_": "MD", "dep_": "aux"},
    {"tag_": "RB", "dep_": "advmod"},
    {"tag_": "JJ", "dep_": "amod"},
]


def check_condition(token):
    return any(
        all(token.__getattribute__(k) == v for k, v in item.items()) for item in conditions
    )


def main(ifilename, opath):
    nlp = spacy.load("en_core_web_trf")

    dfw = pd.read_csv(ifilename)
    acc = []
    for ii, row in dfw.iterrows():
        phrase = row["conclusion_sentence"]
        rdoc, nx_graph = phrase_to_deptree(nlp, phrase)
        redux_phrase = []
        modifiers = []
        for token in rdoc:
            if check_condition(token):
                modifiers += [(row["pmid"], token.dep_, token.tag_, token.lower_)]
            else:
                redux_phrase += [token]
        acc.extend(modifiers)
    dfr = pd.DataFrame(acc, columns=["pmid", "dep", "tag", "text"])
    dfr.to_csv(opath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-csv", help="input csv")
    parser.add_argument("-o", "--output-csv", help="output csv")

    args = parser.parse_args()

    main(os.path.expanduser(args.input_csv), os.path.expanduser(args.output_csv))
