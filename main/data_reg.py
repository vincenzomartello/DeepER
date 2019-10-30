# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:44:07 2019

@author: Giulia
"""
import csv
import random
from itertools import islice 
from sklearn.metrics.pairwise import cosine_similarity
# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from experimental_similarity import mono_vector0,mono_vector,mono_vector_fast,cosine_similarity_vector,distance_similarity_vector
#### WARNING
import re, math
from collections import Counter
from pyjarowinkler import distance
import textdistance
import matplotlib.pyplot as plt
from DeepER import init_embeddings_index
from generate_similarity_vector import wa_vector
import textdistance as txd
import numpy as np
from similarity.metric_lcs import MetricLCS



# ------------------------------ ACM - DBLP2 (LIBRI) - uguale a DBPL-Scholar e CITATIONS -----------------------------------------------
#FLAG= 'BOOKS'

# TUPLA=                        [title, authors, venue, year]
# per ACM.csv colonne:          [  1,      2,      3,     4 ]
# per DBLP2.csv colonne:        [  1,      2,      3,     4 ]

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

##secondo metodo per il calcolo del cos-sim NON UTILIZZATO
def cos_sim2Str(str1,str2):
    documents=[str1,str2]
    count_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b",stop_words=None)
    sparse_matrix = count_vectorizer.fit_transform(documents)
    # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
    doc_term_matrix = sparse_matrix.todense()
    df = pd.DataFrame(doc_term_matrix,columns=count_vectorizer.get_feature_names(),index=['str1', 'str2'])
    #print(df)
    cos_sim=cosine_similarity(df,df)
    #print(cos_sim[0][-1])
    return cos_sim[0][-1]

"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''parsing dei csv e costruzione dataset alternato(match-NOmatch) =>  (tupla1,tupla2,vettore_sim,label_match_OR_no_match)
    indici= lista di Coppie di attributi considerati   (es: per Walmart Amazon (Walmart_att, Amazon_att))
    cosi ogni coppia di tuple ha stesso num di attributi
    ES:   indici=[(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]'''


# calcolo similarità tra numeri normalizzata, basata sulla differenza tra i valori (price o year)
def numeric_normalized_similarity(x,y):
    
    max = 0
    min = 0
    if x >= y:
        max = x
        min = y
    else :
        max = y
        min = x
    
    dif = max - min             
    perc = dif / max
    num = 1.0 - perc

    return num

def books_vector_cosenb(tuple1,tuple2,embeddings_index):
     cos_vect1=cosine_similarity_vector(tuple1, tuple2,embeddings_index)
     return cos_vect1
def books_vector_disenb(tuple1,tuple2,embeddings_index):
    ds_vector1=distance_similarity_vector(tuple1, tuple2,embeddings_index)
    return ds_vector1

# genera vettore di similarita per dataset in formato BOOKS
def books_vector(tuple1, tuple2):

#    jawi = distance.get_jaro_distance(tuple1[0], tuple2[0], winkler=True, scaling=0.1)   # jaro-winkler similarity
#    jacc = textdistance.jaccard.normalized_similarity(tuple1[1], tuple2[1])              # jaccard similarity
#    cos = textdistance.cosine.normalized_similarity(tuple1[2], tuple2[2])               # cosine similarity
#
#    # calcolo similarità tra anni normalizzata, i valori in input saranno stringhe
#    year = 0.0
#
#    try:
#        year = numeric_normalized_similarity(float(tuple1[3]),float(tuple2[3]))
#    except ValueError as val_error:
#        # could not convert string to float
#        pass
#
#
#    concat1 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1], tuple2[0]+' '+tuple2[1])
#    concat2 = textdistance.cosine.normalized_similarity(tuple1[2]+' '+str(tuple1[3]), tuple2[2]+' '+str(tuple2[3]))
#    concat3 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
#                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))

    jawi = distance.get_jaro_distance(tuple1[1], tuple2[1], winkler=True, scaling=0.1)
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])           # levenshtein similarity
    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[0], tuple2[0])        # sorensen-dice similarity
#    w1 = (0.7*jawi) + (0.3*jacc)                                                         # weighed similarity 1
#    w2 = (0.7*lev) + (0.3*sodi)                                                          # weighed similarity 2
    
    
    
    #mono_vect=mono_vector_fast(tuple1, tuple2)
    
    
#    cos_vect1=cosine_similarity_vector(tuple1, tuple2,embeddings_index)
#    ds_vector1=distance_similarity_vector(tuple1, tuple2,embeddings_index)

    
#    vector = [float('%.3f'%(jawi)), float('%.3f'%(jacc)), float('%.3f'%(cos)), float('%.3f'%(year)), 
#                float('%.3f'%(concat1)), float('%.3f'%(concat2)), float('%.3f'%(concat3)),
#                float('%.3f'%(lev)),float('%.3f'%(sodi))]
#                float('%.3f'%(w1)),float('%.3f'%(w2))]  
    vector = [ float('%.3f'%(jawi)),float('%.3f'%(lev)),float('%.3f'%(sodi))]
    #vector.append(mono_vect[0])

    
    return vector

def books_vectorRed(tuple1, tuple2):


    jawi = distance.get_jaro_distance(tuple1[1], tuple2[1], winkler=True, scaling=0.1)
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])           # levenshtein similarity
    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[0], tuple2[0])        # sorensen-dice similarity
    
    vector = [ float('%.3f'%(jawi)),float('%.3f'%(lev)),float('%.3f'%(sodi))]
    
    
    return vector

def books_vector3(tuple1, tuple2):


    jawi = distance.get_jaro_distance(tuple1[1], tuple2[1], winkler=True, scaling=0.1)
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])
    lev1 = textdistance.levenshtein.normalized_similarity(tuple1[2], tuple2[2])           # levenshtein similarity
    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[0], tuple2[0])        # sorensen-dice similarity
   
    
    vector = [ float('%.3f'%(jawi)),float('%.3f'%(lev)),float('%.3f'%(lev1)),
              float('%.3f'%(sodi))]
    
    return vector

def books_vector4(tuple1, tuple2):
    
#    jawi=txd.jaro_winkler.normalized_similarity(tuple1[i], tuple2[i])
#    #jawi = distance.get_jaro_distance(tuple1[i], tuple2[i], winkler=True, scaling=0.1)
#    #lev = textdistance.levenshtein.normalized_similarity(tuple1[i], tuple2[i])
#    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[i], tuple2[i])
#    cos = textdistance.cosine.normalized_similarity(tuple1[i], tuple2[i]) 
#    jacc = textdistance.jaccard.normalized_similarity(tuple1[1], tuple2[1])
    
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])
    jawi=txd.jaro_winkler.normalized_similarity(tuple1[1], tuple2[1])
    jacc = textdistance.jaccard.normalized_similarity(tuple1[2], tuple2[2])

    vector = [ float('%.3f'%(lev)),float('%.3f'%(jawi)),float('%.3f'%(jacc))]
    
    return vector

def books_vector5(tuple1, tuple2):
    
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])
        
    concat3 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))
    mono=mono_vector_fast(tuple1, tuple2)
    vector = [ float('%.3f'%(lev)),float('%.3f'%(concat3))]
    vector.append(mono[0])
    
    return vector


def books_vector6(tuple1, tuple2):
    
#    jawi=txd.jaro_winkler.normalized_similarity(tuple1[i], tuple2[i])
#    #jawi = distance.get_jaro_distance(tuple1[i], tuple2[i], winkler=True, scaling=0.1)
#    #lev = textdistance.levenshtein.normalized_similarity(tuple1[i], tuple2[i])
#    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[i], tuple2[i])
#    cos = textdistance.cosine.normalized_similarity(tuple1[i], tuple2[i]) 
#    jacc = textdistance.jaccard.normalized_similarity(tuple1[1], tuple2[1])
    
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])
    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[0], tuple2[0])
    sodi1 = textdistance.sorensen_dice.normalized_similarity(tuple1[1], tuple2[1])
    jacc = textdistance.jaccard.normalized_similarity(tuple1[2], tuple2[2])
    cos = textdistance.cosine.normalized_similarity(tuple1[3], tuple2[3])

    vector = [ float('%.3f'%(lev)),float('%.3f'%(sodi)),
              float('%.3f'%(sodi1)),float('%.3f'%(jacc)),
              float('%.3f'%(cos))]
    
    return vector


def books_vector10(tuple1, tuple2):
    
    concat3 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))
    mono=mono_vector(tuple1, tuple2)
    vector = [float('%.3f'%(concat3))]
    vector.append(mono[0])
    
    return vector

def books_vector11(tuple1, tuple2):
    
    concat3 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))
    mono=mono_vector_fast(tuple1, tuple2)
    vector = [float('%.3f'%(concat3))]
    vector.append(mono[0])
    
    return vector

def books_vector12(tuple1,tuple2):
   
    lev = textdistance.levenshtein.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))
    
    mono_vect=mono_vector(tuple1, tuple2)
 
    vector = [float('%.3f'%(lev))]  
    vector.append(mono_vect[0])
    
    return vector

def books_vector13(tuple1,tuple2):
    str1 = ' '.join(tuple1).lower()
    str2 = ' '.join(tuple2).lower()
    
    concat3 = textdistance.jaccard.normalized_similarity(str1,str2)
    
    vector = [float('%.3f'%(concat3))]
    
    simv=mono_vector0(tuple1, tuple2)
    mono_vect=[np.mean(simv)]
    lev=[simv[0]]
    
    vector.append(mono_vect[0])
    vector.append(lev[0])
    
    return vector

def books_vector14(tuple1,tuple2):
      
    simv=mono_vector0(tuple1, tuple2)
    mono_vect=[np.mean(simv)]
    simv4=[simv[0],simv[1],simv[2],simv[3]]
    simv2=[simv[0],simv[1],simv[2],simv[4]]
    mono_sim2=[np.mean(simv2)]
    mono_sim4=[np.mean(simv4)]
#    mono_sim2=mono_sim_fun2(tuple1, tuple2)
#    mono_sim4=mono_sim_fun4(tuple1, tuple2)
    vector = []  
    vector.append(mono_vect[0])
    vector.append(mono_sim2[0])
    vector.append(mono_sim4[0])
    
    
    return vector

def books_vector15(tuple1,tuple2):
      
    simv=mono_vector0(tuple1, tuple2)
    mono_vect=[np.mean(simv)]
    lev=[simv[0]]
    
    vector = []  
    vector.append(mono_vect[0])
    vector.append(lev[0])
    
    return vector

def books_vector_round(tuple1,tuple2):
      
    simv=mono_vector0(tuple1, tuple2)
    mono_vect=[np.mean(simv)]   
    
    return mono_vect

def sim_hamming(tuple1,tuple2):
    t1_concat=tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3])#+' '+str(tuple1[4])+' '+str(tuple1[5])
    t2_concat=tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3])#+' '+str(tuple1[4])+' '+str(tuple1[5])
    
    ham=txd.hamming.normalized_similarity(t1_concat,t2_concat)
    vector=[ham]
    return vector

def sim_jaw(tuple1,tuple2):
    t1_concat=tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3])#+' '+str(tuple1[4])+' '+str(tuple1[5])
    t2_concat=tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3])#+' '+str(tuple1[4])+' '+str(tuple1[5])
    
    jawi=txd.jaro_winkler.normalized_similarity(t1_concat,t2_concat)
    vector=[jawi]
    return vector

def sim_lev(tuple1,tuple2):
    t1_concat=tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3])#+' '+str(tuple1[4])+' '+str(tuple1[5])
    t2_concat=tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3])#+' '+str(tuple1[4])+' '+str(tuple1[5])
    
    lev = textdistance.levenshtein.normalized_similarity(t1_concat,t2_concat)
    vector=[lev]
    return vector   
 
def sim_lcs(tuple1,tuple2):
    t1_concat=tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3])+' '+str(tuple1[4])#+' '+str(tuple1[5])
    t2_concat=tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3])+' '+str(tuple1[4])#+' '+str(tuple1[5])
    metric_lcs = MetricLCS()
    lcs1=1 - metric_lcs.distance(t1_concat,t2_concat)
    vector=[lcs1]
    return vector

def calc_average(data1):
    deeper_data = list(map(lambda q: (q[0], q[1], q[3]), data1))
    
    return average
def round_random_dataset(data1,average):
#    temp_list=[]
    data=[]
#    for i in range(len(data1)):
#        temp_list.append(data1[i][2][0])
#        #print(temp_list)
#    
#    average=(sum(temp_list) / len(temp_list))
    print("##################### average: "+str(average))
    for elem in data1:
        #print(elem)
        if elem[2][0]>average:
            data.append(tuple((elem[0],elem[1],1)))
        else:
            data.append(tuple((elem[0],elem[1],0)))
    
    return data

def round_dataset(data1):
    temp_list=[]
    data=[]
    for i in range(len(data1)):
        temp_list.append(data1[i][2][0])
        #print(temp_list)
    
    average=(sum(temp_list) / len(temp_list))
    print("##################### average: "+str(average))
    for elem in data1:
        if elem[2][0]>average:
            data.append(tuple((elem[0],elem[1],[1],elem[3])))
        else:
            data.append(tuple((elem[0],elem[1],[0],elem[3])))
    
    return data,average

def plotting(result_list,index):
    sim_list=[]
    label_list=[]
    t=[]
    
    sim_listArr=[]
    
    sorted_by_secondEthird = sorted(result_list, key=lambda tup: (tup[3], tup[2][index]))
    for i in range(len(sorted_by_secondEthird)):
        
        sim_list.append(sorted_by_secondEthird[i][2][index])
        label_list.append(sorted_by_secondEthird[i][3])
        t.append(i)
        
    average=(sum(sim_list) / len(sim_list))
    print(average)
    for i in range(len(sim_list)):
        
        if sim_list[i]>=average:
            sim_listArr.append(1)
        else:
            sim_listArr.append(0)
    
    plt.plot(t, label_list, '-b',t, sim_list, '-r')
    plt.ylabel('plot_sim'+str(index))
    plt.show()
    
    plt.plot(t, label_list, '-b',t, sim_listArr, '-r' )
    plt.ylabel('plot_simArr'+str(index))
    plt.show()
            
    #return sim_list, label_list, t, sim_listArr


def plot_graph(result_list):
   
    for j in range(len(result_list[0][2])):
      plotting(result_list,j)

#    
#  
#ground_truth='matches_walmart_amazon.csv'
#tableL='walmart.csv'
#tableR='amazonw.csv'
#indici= [(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]    
##ground_truth='DBLP-Scholar_perfectMapping.csv'
##tableL='DBLP1.csv'
##tableR='Scholar.csv'
##indici=[(1, 1), (2, 2), (3, 3), (4, 4)]
##embeddings_index = init_embeddings_index('glove.6B.50d.txt')
####tot=50
#####csv_2_dataset(ground_truth, tableL, tableR,indici)
####csvTable2datasetRANDOM_minCos(tableL, tableR, tot, indici, 0.3)
#result_list1=csv_2_datasetALTERNATE(ground_truth, tableL, tableR,indici)
#
#ds2reg=parsing2regdata(result_list1)
#write_dataset2csv(ds2reg)
#plot_graph(result_list1)

