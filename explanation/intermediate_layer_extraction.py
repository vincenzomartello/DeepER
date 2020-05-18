#!/usr/bin/python
# -*- coding: utf-8 -*-
import torch
import deepmatcher as dm
from deepmatcher.data import MatchingIterator
import torch
import os
import contextlib
from tqdm import tqdm

# methods to save intermediate layer output/input

def _flat_list(l):
    flat_list = []
    for sublist in l:
        for item in sublist:
            flat_list.append(int(item))
    return flat_list


def return_layer_input_output_for_batch(model,hook,batch):
    out = model(batch)
    return (hook.input,hook.output)


def _return_layer_input_for_batch(model,hook,batch):
    out = model(batch)
    return hook.input


current_layer_inputs = []


def _return_input(module,module_input,module_output):
    if isinstance(module_input,tuple):
        current_layer_inputs.append(module_input[0].detach().requires_grad_())
    else:
        current_layer_inputs.append(module_input[0].detach().requires_grad_())

    
def _clearLayerInputsList():
    global current_layer_inputs
    current_layer_inputs = []
    

def return_layer_input(model,layer,dataset_dir,dataset_name,trueLabel,batch_size=32,
                       device ='cuda',ignore_columns=['label']):
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            dataset = dm.data.process_unlabeled(os.path.join(dataset_dir,dataset_name+'.csv'),model\
                                            ,ignore_columns=ignore_columns)
            hook = layer.register_forward_hook(_return_input)
            predictions = model.run_prediction(dataset,batch_size=batch_size)
    ##map in which for each sample id we have the corrisponde vector in intermediate layer
    res = {}
    j = 0
    sample_indexes = predictions.index
    for batch in current_layer_inputs:
        for sample in batch:
            if round(predictions.iloc[j]['match_score'])==trueLabel:
                res[sample_indexes[j]] = sample
            j += 1
    hook.remove()
    _clearLayerInputsList()
    return res


def _return_input_output(module,module_input,module_output):
    global current_layer_input
    global current_layer_output
    if isinstance(module_input,tuple):
        current_layer_input = (module_input[0].detach().requires_grad_())
    else:
        current_layer_input = (module_input[0].detach().requires_grad_())
    if isinstance(module_output,tuple):
        current_layer_output = (module_output[0].detach().requires_grad_())
    else:
        current_layer_output = (module_output.detach().requires_grad_())


def return_layer_input_output(dataset_dir,dataset_name,batch_size,model,layer,device='cuda'):
    dataset = dm.data.process(path=dataset_dir,train=dataset_name+'.csv',left_prefix='ltable_',right_prefix='rtable_',cache=dataset_name+'.pth')
    dataset_tuple = dataset,
    splits = MatchingIterator.splits(dataset_tuple,batch_size=32, device = device)
    tupleids = []
    layer_inputs = []
    layer_outputs = []
    hook = layer.register_forward_hook(_return_input_output)
    for batch in splits[0]:
        tupleids.append(batch.id)
        model(batch)
        layer_inputs.append(current_layer_input)
        layer_outputs.append(current_layer_output)
    id_flattened = list(map(int,_flat_list(tupleids)))
    ##map in which for each sample id we have the corrisponde vector in intermediate layer
    res = {}
    j = 0
    for batch1,batch2 in zip(layer_inputs,layer_outputs):
        for inp,out in zip(batch1,batch2):
            res[id_flattened[j]] = (inp,out)
            j += 1
    hook.remove()
    return res

