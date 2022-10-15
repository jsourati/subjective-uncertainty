import pandas as pd
import argparse
import os


def main(dpath):
    origin = "gw"

    labeled_batch_0 = os.path.join(dpath, "gw_subset_labeled_ABJS.tsv")
    labeled_batch_1 = os.path.join(dpath, "gw_AB_batch2.csv")

    df_0 = pd.read_csv(labeled_batch_0, sep="\t")

    df_1 = pd.read_csv(labeled_batch_1)

    df_gt = pd.read_csv(os.path.join(dpath, f"augmented_{origin}.csv.gz"), index_col=0)

    mapping = {5: 3, 4: 3, 3: 2, 2: 1, 1: 1}
    df_0["label_AB_3"] = df_0["label_AB"].apply(lambda x: mapping[x])

    dfgwo = pd.read_csv(f"~/data/kl/external/2021-10-13/gw_v1.csv.gz", index_col=0)

    df_gw_id = pd.read_csv(
        "~/data/kl/certainty/to_label/gw_subset_tolabel.tsv", sep="\t"
    )
    df_0["pmid"] = df_gw_id["pmid"]

    dfr = pd.merge(df_0, dfgwo, on="pmid", how="inner")
    dfr = pd.merge(df_1, dfgwo, on="pmid", how="inner")

    dfr["correct"] = 1.0 - abs(dfr.pos - dfr.cdf_exp)

    dfr = dfr.loc[dfr["ungrammatical"].isnull()]

    corr = dfr[["label_AB"] + ["correct"]].corr()
    print(dfr.loc[dfr["label_AB"] > 1][["label_AB", "correct"]].corr().iloc[-1])
    print(dfr.loc[dfr["label_AB"] < 3][["label_AB", "correct"]].corr().iloc[-1])
    print(corr)

    # corr = dfr[['label_AB', 'label_JS', 'label_AB_3'] + ["correct"]].corr()
    # print(corr)
    #
    # print(dfr.loc[dfr["label_AB_3"] > 1][['label_AB', 'label_AB_3', "correct"]].corr().iloc[-1])
    # print(dfr.loc[dfr["label_AB_3"] < 3][['label_AB', 'label_AB_3', "correct"]].corr().iloc[-1])
    # print(dfr.loc[dfr["label_JS"] > 1][["label_JS", "correct"]].corr().iloc[0, 1])
    # print(dfr.loc[dfr["label_JS"] < 3][["label_JS", "correct"]].corr().iloc[0, 1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datapath", help="path to labeled data")

    args = parser.parse_args()

    main(args.datapath)
