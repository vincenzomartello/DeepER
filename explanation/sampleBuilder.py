import pandas as pd
import numpy as np
import random as rd
import os
import nltk
from nltk.corpus import stopwords
import string
from .dataset_parser import generateDataset
from strsimpy.jaccard import Jaccard
import math
from tqdm import tqdm



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
    ##remove comment for reproducibility
    ##rd.seed(0)
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


def generateNewNegatives(df,source1,source2,newNegativesToBuild):
    allNewNegatives = []
    jaccard = Jaccard(3)
    df_c = df.copy()
    df_c['ltable_id'] = list(map(lambda lrid:lrid.split("#")[0],df_c.id.values))
    df_c['rtable_id'] = list(map(lambda lrid:lrid.split("#")[1],df_c.id.values))
    positives = df_c[df_c.label==1]
    negatives = df_c[df_c.label==0]
    newNegativesPerSample = math.ceil(newNegativesToBuild/len(positives))
    for i in tqdm(range(len(positives))):
        locc = np.count_nonzero(negatives.ltable_id.values==positives.iloc[i]['ltable_id'])
        rocc = np.count_nonzero(negatives.rtable_id.values == positives.iloc[i]['rtable_id'])
        if locc==0 and rocc == 0:
            permittedIds = [sampleid for sampleid in df_c['rtable_id'].values if sampleid!= df_c.iloc[i]['rtable_id']]
            newNegatives_l = buildNegativeFromSample(positives.iloc[i]['ltable_id'],permittedIds,\
                                                     newNegativesPerSample,source1,source2,jaccard,0.5)
            newNegatives_df = pd.DataFrame(data=newNegatives_l,columns=['ltable_id','rtable_id','label'])
            allNewNegatives.append(newNegatives_df)
    allNewNegatives_df = pd.concat(allNewNegatives)
    return allNewNegatives_df


def prepareDataset(dataset_dir,dataset_filename,source1,source2,newNegativesToBuild,lprefix='ltable_',rprefix='rtable_'):
    dataset = pd.read_csv(os.path.join(dataset_dir,dataset_filename))
    colForDrop = [col for col in list(dataset) if col not in ['id','label']]
    dataset = dataset.drop_duplicates(colForDrop)
    
    source1_df = pd.read_csv(os.path.join(dataset_dir,source1),dtype=str)
    source2_df = pd.read_csv(os.path.join(dataset_dir,source2),dtype=str)
    newNegatives_ids = generateNewNegatives(dataset,source1_df,source2_df,newNegativesToBuild)
    tmp_name = "./{}.csv".format("".join([rd.choice(string.ascii_lowercase) for _ in range(10)]))
    newNegatives_ids.to_csv(os.path.join(dataset_dir,tmp_name),index=False)
    newNegatives_df = generateDataset(dataset_dir,source1,source2,tmp_name,lprefix,rprefix)
    augmentedData = pd.concat([dataset,newNegatives_df],ignore_index=True)
    os.remove(os.path.join(dataset_dir,tmp_name))
    return augmentedData

