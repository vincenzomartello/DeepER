import math 
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from tqdm import tqdm
import pandas as pd
import torch.nn as nn


def _saveCurrentGradient(grad):
    global currentGradient
    currentGradient = grad


def get_probabilites(vec):
    probabilities = F.softmax(vec,dim=1)
    return probabilities


def _getPrediction(layer,sample):
    pred = layer.forward(sample)
    if pred[0][0].item() > pred[0][1].item():
        return 0
    else:
        return 1


def findPerturbationToFlipPredict(sample,layer,classifier_length,attributes,attribute_length,true_label, max_iterations = 100):
    ##to not alter original starting sample
    sample_copy = torch.unsqueeze(sample.clone().detach().requires_grad_(),0)
    xi = sample_copy
    sum_ri = torch.zeros(classifier_length,device='cuda')
    iterations = 1
    continue_search = True
    while(continue_search and iterations <max_iterations and _getPrediction(layer,xi)==true_label):      
        output = layer(xi)
        probabilities = get_probabilites(output)
        hook = xi.register_hook(_saveCurrentGradient)
        ##f(x) is the probability of the current state
        if true_label == 0:
            fx = 1 - probabilities[0][1]
        else:
            fx = probabilities[0][1]
        #- to move to the opposite direction of gradients
        loss = -nn.CrossEntropyLoss()(F.softmax(output,dim=1),torch.tensor([true_label],device='cuda:0'))
        loss.backward()
        current_gradient = currentGradient
        partial_derivative = torch.zeros(classifier_length,device='cuda')
        for att in attributes:
            start_index = att * attribute_length
            end_index = start_index+ attribute_length
            partial_derivative[start_index:end_index] = current_gradient[:,start_index:end_index]
        current_norm = torch.norm(partial_derivative)
        if current_norm.item() <=0.00001:
            sum_ri = torch.zeros(classifier_length,device='cuda')
            print(" Gradient is null")
            continue_search = False
        else:
            ri = -(fx/current_norm)*(partial_derivative)
            xi = (xi+ri).clone().detach().requires_grad_()
            sum_ri += ri
            iterations +=1
            hook.remove()
    if iterations>=max_iterations:
        sum_ri = torch.zeros(classifier_length,device='cuda')
        print("can't converge in {} iterations".format(str(iterations)))
    return sum_ri


def computeRi(layer,attributes,dataset,true_label,aggregation_type):
    layer_len = len(list(dataset.values())[0])
    attribute_len = int(layer_len/len(attributes))
    ri_map = {}
    i = 0
    for sampleid in tqdm(dataset.keys()):
        sample = dataset[sampleid]
        current_sample_ris = list(map(lambda att: findPerturbationToFlipPredict(sample,layer,layer_len,[attributes.index(att)],
                                                                                         attribute_len,true_label),attributes))
        ri_map[sampleid] = current_sample_ris
    aggregatedRi = aggregateRi(ri_map,dataset,attributes,aggregation_type)
    return ri_map,aggregatedRi


def aggregateRi(ri_map,samples,attributes,aggregation_type):
    aggregation = []
    attribute_len = int(len(list(samples.values())[0])/len(attributes))
    for sample_id in samples.keys():
        curr_aggregate = []
        for idx,_ in enumerate(attributes):
            start_idx = idx*attribute_len
            end_idx = start_idx+attribute_len
            curr_ri = ri_map[sample_id][idx]
            if aggregation_type=='cosine':
                perturbation_measure = F.cosine_similarity(samples[sample_id][start_idx:end_idx],\
                                                           (samples[sample_id]+curr_ri)[start_idx:end_idx],dim=0)
            elif aggregation_type=='euclidean':
                perturbation_measure = torch.norm(curr_ri)
            curr_aggregate.append(perturbation_measure.item())
        aggregation.append(curr_aggregate)
    df = pd.DataFrame(data=aggregation,columns=attributes)
    df['sample_id'] = samples.keys()
    return df


def findCloserNaif(v,opposite_data,opposite_data_label,model,attribute_idx,attribute_len):
    start_idx = attribute_idx*attribute_len
    end_idx = start_idx+attribute_len
    original_prediction = model.forward(torch.unsqueeze(v,0))
    if original_prediction[0][0].item() > original_prediction[0][1].item():
        original_label = 0
    else:
        original_label = 1
    original_att_value = v[start_idx:end_idx].clone()
    distances = []
    perturbations = []
    for batch in opposite_data:
        len_batch = len(batch)
        v_batch = torch.unsqueeze(v,0).clone()
        v_batch_var = Variable(v_batch.data.repeat(len_batch,1))
        batch_copy = batch.clone()
        v_batch_var[:,start_idx:end_idx] = batch_copy[:,start_idx:end_idx]
        predictions = model.forward(v_batch_var)
        i = 0
        for pred in predictions:
            if (opposite_data_label ==1 and pred.data[1]>pred.data[0]) or (opposite_data_label ==0 \
                                                                           and pred.data[0] >pred.data[1]):
                ri = batch[i][start_idx:end_idx]-original_att_value
                distances.append(torch.norm(ri).data[0])
                perturbations.append(batch[i][start_idx:end_idx])
            i +=1
    if len(distances) >0 and original_label != opposite_data_label:
        closer_distance = min(distances)
        best_idx = distances.index(closer_distance)
        return (perturbations[best_idx])
    else:
        ##If we return 0 vector it means that we couldn't find any perturbation
        return Variable(torch.zeros(attribute_len))


def computeRiNaif(dataset,oppLabelData,oppLabel,layer,attributes,attribute_len):
    ri = []
    for batch in dataset:
        for sample in tqdm(batch):
            currentRis = list(map(lambda att : findCloserNaif(sample,oppLabelData,oppLabel,layer,
                                                              attributes.index(att),attribute_len),attributes))
            ri.append(currentRis)
    ri_norms = [[torch.norm(v).data[0] for v in ris] for ris in ri]
    return ri,ri_norms