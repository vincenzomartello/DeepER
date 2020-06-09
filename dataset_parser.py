import pandas as pd
import os
import random
import numpy as np


def generate_train_valid_test(dataset_dir,source1,source2,splitfiles,lprefix,rprefix):
    datasets = []
    for split in splitfiles:
        datasets.append(generateDataset(dataset_dir,source1,source2,split,lprefix,rprefix))
    return datasets[0],datasets[1],datasets[2]


def generateDataset(dataset_dir,source1,source2,pairs_ids,lprefix,rprefix):
    source1_df = pd.read_csv(os.path.join(dataset_dir,source1))
    source2_df = pd.read_csv(os.path.join(dataset_dir,source2))
    pairs_ids_df = pd.read_csv(os.path.join(dataset_dir,pairs_ids))
    ##to avoid duplicate columns
    pairs_ids_df.columns = ['id1','id2','label']
    lcolumns,rcolumns = ([],[])
    for lcol,rcol in zip(list(source1_df),list(source2_df)):
        lcolumns.append(lprefix+lcol)
        rcolumns.append(rprefix+rcol)
    source1_df.columns = lcolumns
    source2_df.columns = rcolumns
    pdata = pd.merge(pairs_ids_df,source1_df, how='inner',left_on='id1',right_on=lprefix+'id')
    dataset = pd.merge(pdata,source2_df,how='inner',left_on='id2',right_on=rprefix+'id')
    dataset[lprefix+'id'] = dataset[lprefix+'id'].astype(str)
    dataset[rprefix+'id'] = dataset[rprefix+'id'].astype(str)
    dataset['id'] = "0@"+dataset[lprefix+'id']+"#"+"1@"+dataset[rprefix+'id']
    dataset = dataset.drop(['id1','id2',lprefix+'id',rprefix+'id'],axis=1)
    return dataset


def get_pos_neg_datasets(splits):
    allSamples = pd.concat(splits,ignore_index=True)
    positives_df = allSamples[allSamples.label==1]
    negatives_df = allSamples[allSamples.label==0]
    return positives_df,negatives_df


def generate_unlabeled(dataset_dir,unlabeled_filename,lprefix='ltable_',rprefix='rtable_'):
    df_tableA = pd.read_csv(os.path.join(dataset_dir,'tableA.csv'),dtype=str)
    df_tableB = pd.read_csv(os.path.join(dataset_dir,'tableB.csv'),dtype=str)
    unlabeled_ids = pd.read_csv(os.path.join(dataset_dir,unlabeled_filename),dtype=str)
    left_columns = list(map(lambda s:lprefix+s,list(df_tableA)))
    right_columns = list(map(lambda s:rprefix+s,list(df_tableB)))
    df_tableA.columns = left_columns
    df_tableB.columns = right_columns

    unlabeled_df = unlabeled_ids.merge(df_tableA, how='inner',left_on=lprefix+'id',\
                                     right_on='ltable_id').merge(df_tableB,how='inner',left_on='rtable_id',right_on=rprefix+'id')
    unlabeled_df['id'] = unlabeled_df[lprefix+'id']+"#"+unlabeled_df[rprefix+'id']
    unlabeled_df = unlabeled_df.drop(['ltable_id','rtable_id'],axis=1)
    return unlabeled_df


def getFullDataset(splits):
    return pd.concat(splits,ignore_index=True)



def dropTokensInColumns(df,attributes,tokensL):
    df_copy = df.copy()
    for attr in attributes:
        df_copy[attr] = df_copy[attr].apply(lambda r:dropTokens(r,tokensL))
    return df_copy


def dropTokens(attr,tokensL):
    attr_tokens = list(map(lambda t:t.lower(),attr.split()))
    filtered_tokens = []
    for tok in attr_tokens:
        if tok not in tokensL:
            filtered_tokens.append(tok)
    return " ".join(filtered_tokens)

