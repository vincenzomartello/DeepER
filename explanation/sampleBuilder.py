#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random as rd
import nltk
from nltk.corpus import stopwords
import string


# #df1 and df2 are two sources of records, attributes are the attributes important for the prediction
def createPossibleMatchings(
    df1,
    df2,
    important_attributes,
    min_similarity,
    newsample_len,
    comparator,
    leftPrefix='ltable_',
    rightPrefix='rtable_'
    ):
    df1_index = list(df1.index)
    df2_index = list(df2.index)
    newSamples = []
    while len(newSamples) < newsample_len:
        current_lrow = df1.iloc[rd.choice(df1.index)]
        current_df = df2
        for att in important_attributes:
            l_att = str(current_lrow[att])
            mask = current_df.apply(lambda row: \
                                    comparator.similarity(str(row[att]),
                                    l_att) >= min_similarity, axis=1)
            current_df = current_df[mask]
        for idx in current_df.index:
            newSamples.append((current_lrow['id'], current_df.at[idx,
                              'id']))
    newSamples_ids = pd.DataFrame(data=newSamples, columns=['ltable_id'
                                  , 'rtable_id'])

        # #only temporary label

    newSamples_ids['label'] = np.ones(newSamples_ids.shape[0])
    left_columns = list(map(lambda s: leftPrefix + s, list(df1)))
    right_columns = list(map(lambda s: rightPrefix + s, list(df2)))
    df1.columns = left_columns
    df2.columns = right_columns

        # P sta per parziale

    punlabeled = pd.merge(newSamples_ids, df1, how='inner')
    unlabeled_df = pd.merge(punlabeled, df2, how='inner')

    unlabeled_df = unlabeled_df.drop(['ltable_id', 'rtable_id'], axis=1)
    unlabeled_df['id'] = np.arange(unlabeled_df.shape[0])
    return (unlabeled_df, newSamples_ids)


    # #new attribute value is a couple of attributes

def buildNewSamples(
    dataset,
    selectedAttr,
    newAttributeVal,
    newSamples_len,
    label,
    left_prefix='ltable_',
    right_prefix='rtable_',
    ):

    new_samples = pd.DataFrame(data=[], columns=list(dataset))
    for i in range(newSamples_len):
        selected_row = dataset.sample()
        if label == 0:
            selected_row[left_prefix + selectedAttr] = \
                newAttributeVal[0]
            selected_row[right_prefix + selectedAttr] = \
                newAttributeVal[1]
        else:
            selected_row[left_prefix + selectedAttr] = \
                selected_row[left_prefix + selectedAttr] + ' ' \
                + newAttributeVal[0]
            selected_row[right_prefix + selectedAttr] = \
                selected_row[right_prefix + selectedAttr] + ' ' \
                + newAttributeVal[1]
        new_samples = new_samples.append(selected_row,
                ignore_index=True)
    return new_samples


def buildNewSamplesForAttribute(
    critical_forPos,
    critical_forNeg,
    attribute,
    lenNewPositives,
    lenNewNegatives,
    start_idx,
    ):
    newSamples = []
    for (df, _, _) in critical_forPos[attribute]:
        if df.shape[0] < lenNewPositives:
            newSamples.append(df)
        else:
            newSamples.append(df.sample(n=lenNewPositives))
    for (df, _, _) in critical_forNeg[attribute]:
        if df.shape[0] < lenNewNegatives:
            newSamples.append(df)
        else:
            newSamples.append(df.sample(n=lenNewNegatives))
    newSamples = pd.concat(newSamples)
    newSamples = newSamples.drop(columns=['match_score'])
    newSamples['id'] = np.arange(start_idx, start_idx
                                 + newSamples.shape[0])
    return newSamples


def buildNegativeFromSample(
    sampleid,
    permittedIds,
    nsamplestobuild,
    source1,
    source2,
    comparator,
    threshold
    ):
    ## for reproducibility
    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    attributes = [att for att in list(source1) if att not in ['id']]
    r1 = source1[source1.id==sampleid].iloc[0]
    if nsamplestobuild > len(permittedIds):
        nsamplestobuild = len(permittedIds)
    r2_ids = set()
    alreadyConsideredIds = []
    while((len(r2_ids)<nsamplestobuild)or(len(alreadyConsideredIds)<nsamplestobuild)):
        r2_id = rd.choice(permittedIds)
        while (r2_id in alreadyConsideredIds):
            r2_id = rd.choice(permittedIds)
        alreadyConsideredIds.append(r2_id)
        r2 = source2[source2.id==r2_id].iloc[0]
        avg_similarity = 0
        for attr in attributes:
            r1_attr_tokens = nltk.word_tokenize(str(r1[attr]))
            r2_attr_tokens = nltk.word_tokenize(str(r2[attr]))
            r1_clean_tokens = [t for t in r1_attr_tokens if t not in stop_words and t not in punctuation]
            r2_clean_tokens = [t for t in r2_attr_tokens if t not in stop_words and t not in punctuation]
            r2_attr_str = " ".join(r2_clean_tokens)
            r1_attr_str = " ".join(r1_clean_tokens)
            currentSimilarity = comparator.similarity(r1_attr_str,r2_attr_str)
            avg_similarity += currentSimilarity
        if (avg_similarity/len(attributes)) < threshold:
            r2_ids.add(r2['id'])
    negativePairs = []
    for r2_id in r2_ids:
        negativePairs.append((r1['id'],r2_id,0))
    return negativePairs