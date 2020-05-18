import deepmatcher as dm
import numpy as np
import contextlib
import pandas as pd
from tqdm import tqdm
import random 
import os
import string
from sklearn.metrics import f1_score,precision_score,recall_score


def wrapDm(test_df,model,ignore_columns=['id','label'],outputAttributes=False,batch_size=32):
    data = test_df.copy().drop([c for c in ignore_columns if c in test_df.columns],axis=1)
    data['id'] = np.arange(len(test_df))
    tmp_name = "./{}.csv".format("".join([random.choice(string.ascii_lowercase) for _ in range(10)]))
    data.to_csv(tmp_name,index=False)
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            data_processed = dm.data.process_unlabeled(tmp_name, trained_model = model)
            predictions = model.run_prediction(data_processed, output_attributes = outputAttributes,\
                                              batch_size=batch_size)
            out_proba = predictions['match_score'].values.reshape(-1)
    multi_proba = np.dstack((1-out_proba, out_proba)).squeeze()
    os.remove(tmp_name)
    if outputAttributes:
        return predictions
    else:
        return multi_proba



def getF1PrecisionRecall(true_labels,predicted_labels):
    y_pred = np.argmax(predicted_labels,axis=1)
    return (f1_score(true_labels,y_pred),precision_score(true_labels,y_pred),
            recall_score(true_labels,y_pred))


def getMeanConfidenceAndVariance(model,test_df,ignoreColumns=['id','label']):
    predictions = wrapDm(test_df,model,ignore_columns=ignoreColumns)
    confidences = np.amax(predictions,axis=1)
    meanConfidence = sum(confidences)/len(confidences)
    variance = np.var(confidences)
    return meanConfidence,variance


def getTruePositiveNegative(model,df,ignore_columns):
    predictions = wrapDm(df,model,ignore_columns=ignore_columns)
    tp_group = df[(predictions[:,1]>=0.5)& df['label'] == 1]
    tn_group = df[(predictions[:,0] >=0.5)& df['label']==0]
    correctPredictions = pd.concat([tp_group,tn_group])
    return correctPredictions