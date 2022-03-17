import pandas as pd
from subj_unc.models import SubjectiveUncertainty
from subj_unc.utils import measure_uncertainty_df_abstracts
import logging
import argparse
import sys


def main():
    df = pd.read_csv("~/data/kl/certainty/v2/augmented_gw_v1.csv.gz")
    df = df.loc[df["abstract"].notnull()].copy().head(1000)
    m = SubjectiveUncertainty("zero-shot")
    dft = measure_uncertainty_df_abstracts(df, m, 5)
    dft.to_csv("~/data/kl/certainty/v5/augmented_gw.csv.gz")
    print(f"{type(m)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filemode="w",
        stream=sys.stdout,
    )
    #
    # parser.add_argument("--config-path", type=str, help="path to yaml config")

    main()
