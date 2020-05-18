from .intermediate_layer_extraction import return_layer_input
from .ri_calculator import computeRi
from .dataset_parser import generate_train_valid_test
import pandas as pd
import os
from .mojito2 import explainSamples


class Explainer:
    def __init__(self,model,attributes):
        self.model = model
        self.attributes = attributes

    
    def getRankingsWhiteBox(self,dataset_dir,dataset_filename,true_label,aggregation_type):
        print('Computing vectors in the classifier space')
        vectors = return_layer_input(self.model,self.model.classifier,dataset_dir,dataset_filename,true_label)
        ri,ri_aggregate = computeRi(self.model.classifier,\
            self.attributes,vectors,true_label,aggregation_type=aggregation_type)
        return ri,ri_aggregate,vectors


    def getRankingsBlackBox(self,df,sources,predict_fn,true_label,max_len_attribute_set):
        predictions = predict_fn(df,self.model,['label'])
        tp_group = df[(predictions[:,1]>=0.5)& (df['label'] == 1)]
        tn_group = df[(predictions[:,0] >=0.5)& (df['label']==0)]
        correctPredictions = pd.concat([tp_group,tn_group])
        rankings,triangles,flipped,notFlipped = explainSamples(correctPredictions,sources,\
            self.model,predict_fn,originalClass=true_label,maxLenAttributeSet=max_len_attribute_set)
        return rankings,flipped