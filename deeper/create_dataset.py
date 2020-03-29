import random as rd
import math,re,csv
from collections import Counter
import os
import pandas as pd


WORD = re.compile(r'\w+')

#calcola la cos similarity di due vettori
def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)


def concatenate_list_data(list):
    result= ''
    for element in list:
        result += ' '+str(element)
    return result


def buildDataset_from_aligned(dataset_dir,ground_truth,tableL, tableR,negpos_ratio):
    rd.seed(0)
    table1 = pd.read_csv(os.path.join(dataset_dir,tableL))
    table2 = pd.read_csv(os.path.join(dataset_dir,tableR))
    matches_df = pd.read_csv(os.path.join(dataset_dir,ground_truth))

    result_list_match = []
    result_list_NOmatch= []
    matches_ids = list(map(lambda x, y:(x,y), matches_df.id1.values, matches_df.id2.values))
    nonMatches_ids = []
    cos_sim_list=[]
    for i in range(len(matches_df)):
        curr_lid = matches_df.iloc[i]['id1']
        curr_rid = matches_df.iloc[i]['id2']
        #e' una riga singola
        curr_lrecord = table1[table1.id==curr_lid].iloc[0]
        curr_rrecord = table2[table2.id==curr_rid].iloc[0]
        curr_ltokens = list(map(lambda t:str(t),curr_lrecord.values[1:]))
        curr_rtokens = list(map(lambda t:str(t),curr_rrecord.values[1:]))
        stringa1 = " ".join(curr_ltokens)
        stringa2 = " ".join(curr_rtokens)
        #calcola la cos similarita della tupla i-esima
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        cos_sim_list.append(cos_sim)
        result_list_match.append((curr_ltokens,curr_rtokens,1))
    
    min_cos_sim_match=min(cos_sim_list)
    '''costruisce la lista dei NO_match basandosi sulla min cos similarity dei positivi'''   
    i=0
    while i<len(result_list_match)*negpos_ratio:

        x = rd.randint(0,len(table1)-1)
        y =  rd.randint(0,len(table2)-1)            
        curr_lrecord = table1.iloc[x]
        curr_rrecord = table2.iloc[y]
        curr_ltokens = list(map(lambda t:str(t),curr_lrecord.values[1:]))
        curr_rtokens = list(map(lambda t:str(t),curr_rrecord.values[1:]))
        stringa1 = " ".join(curr_ltokens)
        stringa2 = " ".join(curr_rtokens)
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        
        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim > min_cos_sim_match and (x,y) not in matches_ids:
            result_list_NOmatch.append((curr_ltokens,curr_rtokens,0))
            nonMatches_ids.append((x,y))
            i += 1
            
    matches_ids_df = pd.DataFrame(matches_ids,columns=['ltable_id','rtable_id'])
    nonMatches_ids_df = pd.DataFrame(nonMatches_ids,columns=['ltable_id','rtable_id'])
    matches_ids_df['label'] = [1]*len(matches_ids_df)
    nonMatches_ids_df['label'] = [0]*len(nonMatches_ids_df)
    allSamples_df = pd.concat([matches_ids_df,nonMatches_ids_df],ignore_index=True)
    return allSamples_df
