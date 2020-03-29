from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from itertools import islice


def _take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def countTokens(attributes):
    allTokens = {}
    for attr in attributes:
        tokens = list(map(lambda s:s.lower(),attr.split()))
        for tok in tokens:
            if tok in allTokens:
                allTokens[tok] += 1
            else:
                allTokens[tok] = 1
    allOrderedTokens = OrderedDict(sorted(allTokens.items(), key=lambda t: t[1],reverse=True))
    return allOrderedTokens


def plotMostFrequentTokens(df,alteredAttributes,topn,chartFilename):
    df_filtered = df[df.alteredAttributes==alteredAttributes]
    D = countTokens(df_filtered['rtable_'+alteredAttributes[0]].values)
    topnD = _take(topn,D.items())
    fig,ax = plt.subplots(figsize=(topn,5))
    x = list(map(lambda kv:kv[0],topnD))
    y = list(map(lambda kv:kv[1],topnD))
    ax.bar(x,y,align='center')
    plt.xticks(np.arange(len(x)),x, rotation='vertical')
    plt.savefig(chartFilename)
    plt.show()