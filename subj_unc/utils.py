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

    cats = ["BACKGROUND", "OBJECTIVE", "METHOD", "RESULT", "CONCLUSION"]

    assert category in cats+['ALL'], "Input is not among the available categories."
    
    results = {**{'paperid': []}, **{x: [] for x in cats}}
    tqdm_list = tqdm(range(len(parsed_df)), position=0, leave=True)
    for i in tqdm_list:
        row = parsed_df.iloc[i]
        rtype = row[0]

        if rtype == "ABSTRACT":
            if i > 0:
                # insert the results into the main dictionary
                results['paperid'] += [paperid]
                for cat in cats:
                    results[cat] += [entry_dict[cat].strip()]
            paperid = row[1]
            entry_dict = {x:'' for x in cats}
        elif rtype not in cats:
            # continue if the row-type is not among the possible ones
            # For instance, there might be 'O'
            continue

        else:
            # if category is 'ALL' consider all the statements, and
            # enter them into the place in the result dictionary
            if category == "ALL":
                entry_dict[rtype] += " " + row[1]
            elif rtype == category:
                entry_dict[rtype] += " " + row[1]

    if category != "ALL":
        for cat in set(cats) - {category}:
            del results[cat]

    return results


def measure_uncertainty_df_abstracts(
    df, sub_unc, block_size, text_column="abstract", save_path=None
):
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
    ignored_pids = []
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

        # removing long sentences that are untokenized for any reason
        thr = 1500
        ignored_pids += [x[0] for x in flat_list if len(x[2])>=thr]
        flat_list = [x for x in flat_list if len(x[2])<thr]
        
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
            dfa.to_csv(save_path, mode='a', sep='\t', header=header, index=False)

    if (len(ignored_pids)>0) and (save_path is not None):
        ignored_pids_path = save_path.split('.')[0]+'_ignored_pids.txt'
        np.savetxt(ignored_pids_path, ignored_pids, fmt='%d')
        
            
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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
           the values of the time history of the signal.
           window_size : int
           the length of the window. Must be an odd integer number.
       order : int
           the order of the polynomial used in the filtering.
           Must be less then `window_size` - 1.
       deriv: int
           the order of the derivative to compute (default = 0 means only smoothing)
       Returns
       -------
       ys : ndarray, shape (N)
           the smoothed signal (or it's n-th derivative).
       Notes
       -----
       The Savitzky-Golay is a type of low-pass filter, particularly
       suited for smoothing noisy data. The main idea behind this
       approach is to make for each point a least-square fit with a
       polynomial of high order over a odd-sized window centered at
       the point.
       Examples
       --------
       t = np.linspace(-4, 4, 500)
       y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
       ysg = savitzky_golay(y, window_size=31, order=4)
       import matplotlib.pyplot as plt
       plt.plot(t, y, label='Noisy signal')
       plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
       plt.plot(t, ysg, 'r', label='Filtered signal')
       plt.legend()
       plt.show()
       References
       ----------
       .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
          Data by Simplified Least Squares Procedures. Analytical
          Chemistry, 1964, 36 (8), pp 1627-1639.
       .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
          W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
          Cambridge University Press ISBN-13: 9780521880688
       """
    import numpy as np
    from math import factorial
    
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
