import csv
import random
from itertools import islice 
from sklearn.metrics.pairwise import cosine_similarity
# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import os
#### WARNING
import re, math
from collections import Counter
'''
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!IMPORTANTE!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!IMPORTANTE!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8

"""
si deve fare attenzione all'ordine con cui si passano i table1 e table2
devono essere passati come appaiono nella ground_truth

ha bisogno che i csv dati siano codificati in  utf8
in caso di errore aprirli con textpad->formato->converti in utf8
"""

'''

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

'''parsing del csv e costruzione dataset => (tupla1, tupla2, vettore_sim, label_match_OR_no_match)
    con shuffle finale dei match-No match'''
def csv_2_dataset(dataset_dir,ground_truth, tableL, tableR, indici, sim_function=lambda x, y: [1, 1]):
    
    table1 = csv.reader(open(os.path.join(dataset_dir,tableL),encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(os.path.join(dataset_dir,tableR),encoding="utf8"), delimiter=',')
    matches_file = csv.reader(open(os.path.join(dataset_dir,ground_truth),encoding="utf8"), delimiter=',')
    
    #convert to list for direct access and skip header
    tableLlist = list(table1)[1:]
    tableRlist = list(table2)[1:]
    matches_list = list(matches_file)[1:]

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    #costruisce lista dei match
    for line_in_file in matches_list:
        #line_in_file type: id_1, id_2
        row1=[item for item in tableLlist if item[0]==line_in_file[0]]
        row2=[item for item in tableRlist if item[0]==line_in_file[1]]
        
        #print(row1[0])
        tableL_el=[]
        tableR_el=[]
        for i1,i2 in indici:
            tableL_el.append(row1[0][i1])
            tableR_el.append(row2[0][i2])
        
        
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        #calcola la cos_sim tra le due righe
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #print(cos_sim)
        cos_sim_list.append(cos_sim)
        #calcola il vettore di similarita
        sim_vector=sim_function(tableL_el,tableR_el) # Modificato
        
        result_list_match.append((tableL_el,tableR_el,sim_vector, 1))
        #minimo valore di cos_similarità tra tutte le tuple in match
    min_cos_sim_match=min(cos_sim_list)

    ##[1:] serve per togliere l id come attributo
        
    #costruzione lista dei non match
    i=0
    while i<len(result_list_match):

        x = random.randint(1,len(tableLlist)-1)
        y =  random.randint(1,len(tableRlist)-1)
        
        tableL_el=[]
        tableR_el=[]
                
        for i1,i2 in indici:
            tableL_el.append(tableLlist[x][i1])
            tableR_el.append(tableRlist[y][i2])              
        
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        '''controlla che la tupla che sto aggiungendo abbia una cos_similarity 
        maggiore del min di quelle in match e che non sia nella lista dei match'''
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)
            
            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :
            
                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0))
                i += 1

    #unisce le due liste (match e non match) e fa uno shuffle 
    result_list_match.extend(result_list_NOmatch)
    random.shuffle(result_list_match)    

    return result_list_match    



def csv_2_dataset_aligned(dataset_dir,ground_truth,tableL,tableR,neg_pos_ratio=1):
    
    table1 = csv.reader(open(os.path.join(dataset_dir,tableL),encoding="utf8"), delimiter=',')
    table2 = csv.reader(open(os.path.join(dataset_dir,tableR),encoding="utf8"), delimiter=',')
    matches_file = csv.reader(open(os.path.join(dataset_dir,ground_truth),encoding="utf8"), delimiter=',')
    #convert to list for direct access and skip header
    tableLlist = list(table1)[1:]
    tableRlist = list(table2)[1:]
    matches_list = list(matches_file)[1:]
    ltable_ids = list(map(lambda row:row[0],tableLlist))
    rtable_ids = list(map(lambda row:row[0],tableRlist))
    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
    for line in matches_list:
        #line_in_file type: id_1, id_2
        lrecord= next((record for record in tableLlist if record[0] == line[0]), None)
        rrecord = next((record for record in tableRlist if record[0] == line[1]), None)
        #indice 0 indica l'id, dunque viene escluso
        stringa1= " ".join(lrecord[1:])
        stringa2 = " ".join(rrecord[1:])
        #calcola la cos similarita della tupla i-esima
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        cos_sim_list.append(cos_sim)
        result_list_match.append((lrecord[1:],rrecord[1:],1,line[0]+"#"+line[1]))
    min_cos_sim_match=min(cos_sim_list)
    print("min cosine similarity of matches is {}".format(min_cos_sim_match))
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''   
    i=0
    while i<len(result_list_match*neg_pos_ratio):
        x,y = random.sample(ltable_ids,k=1)[0],random.sample(rtable_ids,k=1)[0]
        lrecord= next((record for record in tableLlist if record[0] == x), None)
        rrecord = next((record for record in tableRlist if record[0] == y), None)
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1= " ".join(lrecord[1:])
        stringa2 = " ".join(rrecord[1:])
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        current_ids = [x,y]
        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match and current_ids not in matches_list:
            result_list_NOmatch.append((lrecord[1:],rrecord[1:],0,x+"#"+y))
            i += 1

    result_list= result_list_match+result_list_NOmatch

    return result_list



def parsing_anhai_dataOnlyMatch(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):
    
    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)
    
    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list_match=[]
    result_list_NOmatch=[]
    result_list=[]
    
    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        if int(line_in_file[2])==1:
                  
            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])
            
            
            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)
            
            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #print(cos_sim)
            sim_vector=sim_function(tableA_el,tableB_el) # Modificato
            cos_sim_list.append(cos_sim)
            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))  
        
        
    min_cos_sim_match=min(cos_sim_list)
    print("min_cos_sim_match"+str(min_cos_sim_match))
    

##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''   
    i=0
    while i<len(result_list_match):

        x = random.randint(1,len(tableAlist)-1)
        y =  random.randint(1,len(tableBlist)-1)
        tableL_el=[]
        tableR_el=[]
                
        for i1,i2 in indici:
            tableL_el.append(tableAlist[x][i1])
            tableR_el.append(tableBlist[y][i2])              
        
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #print(cos_sim)
        
        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)
            
            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :
            
                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0)) 
                i=i+1
    
        '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list

def parsing_anhai_dataOnlyMatch4roundtraining(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):
    
    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)
    
    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list_match=[]
    result_list_NOmatch=[]
    result_list=[]
    
    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        if int(line_in_file[2])==1:
                  
            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])
            
            
            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)
            
            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #print(cos_sim)
            sim_vector=sim_function(tableA_el,tableB_el) # Modificato
            cos_sim_list.append(cos_sim)
            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))  
        
        
    min_cos_sim_match=min(cos_sim_list)
    print("min_cos_sim_match"+str(min_cos_sim_match))
    

##[1:] serve per togliere l id come attributo
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''   
    i=0
    sim_list=[]
    while i<len(result_list_match):
        x = random.randint(1,len(tableAlist)-1)
        y =  random.randint(1,len(tableBlist)-1)
        tableL_el=[]
        tableR_el=[]
                
        for i1,i2 in indici:
            tableL_el.append(tableAlist[x][i1])
            tableR_el.append(tableBlist[y][i2])              
        
        #serve per calcolare la cos_sim tra i due elementi della tupla, è necessario concatenare tutta la riga
        stringa1=concatenate_list_data(tableL_el)
        stringa2=concatenate_list_data(tableR_el)
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #print(cos_sim)
        
        #controlla che la tupla che sto aggiungendo abbia una cos_similarity maggiore del min di quelle in match
        if cos_sim>min_cos_sim_match:
            sim_vector=sim_function(tableL_el,tableR_el)
            
            if (tableL_el,tableR_el,sim_vector,1) not in result_list_match :
                sim_list.append(sim_vector)
                result_list_NOmatch.append((tableL_el,tableR_el,sim_vector,0)) 
                i=i+1
    
        '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])
    average=max(sim_list)

    return result_list,average[0]
    

def parsing_anhai_data(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):
    
    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)
    
    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)
    cos_sim_list=[]
    result_list=[]
    
    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        
        row1=[item for item in tableAlist if item[0]==line_in_file[0]]
        row2=[item for item in tableBlist if item[0]==line_in_file[1]]
        tableA_el=[]
        tableB_el=[]
        for i1,i2 in indici:
            tableA_el.append(row1[0][i1])
            tableB_el.append(row2[0][i2])
        
        
        stringaA=concatenate_list_data(tableA_el)
        stringaB=concatenate_list_data(tableB_el)
        
        #calcola la cos similarita della tupla i-esima
        cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
        #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
        #print(cos_sim)
        sim_vector=sim_function(tableA_el,tableB_el) # Modificato
        if int(line_in_file[2])==1:
            cos_sim_list.append(cos_sim)
            result_list.append((tableA_el,tableB_el,sim_vector, 1))
        else:
            result_list.append((tableA_el,tableB_el,sim_vector, 0))
    
    i=0
    flag=True
    while (i<len(result_list) and flag):
        el=result_list[i][3]
        if int(el)==1:
            first_match=result_list.pop(i)
            result_list.insert(0, first_match)
            flag=False
        else:
            i=i+1
           
    return result_list
    
def ratio_dupl_noDup4Anhai(dataset,index):
    sort_dataset = sorted(dataset, key=lambda tup: (tup[3], tup[2][index]))
    match_number=sum(map(lambda x : x[3] == 1, sort_dataset))
    print("match_number: "+str(match_number))
    n=len(sort_dataset)-(match_number*2)
    print(n)
    listout=sort_dataset[n:]
#    for i in range(105,115):
#        print(listout[i])
    return listout

    
def check_anhai_dataset(ground_truth, tableA, tableB, indici, sim_function=lambda x, y: [1, 1]):
    
    tableA = csv.reader(open(tableA,encoding="utf8"), delimiter=',')
    tableB = csv.reader(open(tableB,encoding="utf8"), delimiter=',')
    trainFile = csv.reader(open(ground_truth,encoding="utf8"), delimiter=',')
    No_match_with_cos_too_small=0
    #skip header
    next(tableA, None)
    next(tableB, None)
    next(trainFile, None)
    
    #convert to list for direct access
    tableAlist = list(tableA)
    tableBlist = list(tableB)
    trainFilelist = list(trainFile)

    result_list_match = []
    result_list_NOmatch= []
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
    for line_in_file in trainFilelist:
        #line_in_file type: id_1, id_2
        
        if int(line_in_file[2])==1:#se è un match
            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])
            
            
            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)
            
            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #print(cos_sim)
            cos_sim_list.append(cos_sim)
            
            sim_vector=sim_function(tableA_el,tableB_el) # Modificato
            
            result_list_match.append((tableA_el,tableB_el,sim_vector, 1))
            #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
    min_cos_sim_match=min(cos_sim_list)
    print("min coseno match:"+str(min_cos_sim_match))
            
    for line_in_file in trainFilelist:
        #else:
        if int(line_in_file[2])==0:
            row1=[item for item in tableAlist if item[0]==line_in_file[0]]
            row2=[item for item in tableBlist if item[0]==line_in_file[1]]
        
            tableA_el=[]
            tableB_el=[]
            for i1,i2 in indici:
                tableA_el.append(row1[0][i1])
                tableB_el.append(row2[0][i2])
            
            
            stringaA=concatenate_list_data(tableA_el)
            stringaB=concatenate_list_data(tableB_el)
            
            #calcola la cos similarita della tupla i-esima
            cos_sim=get_cosine(text_to_vector(stringaA),text_to_vector(stringaB))
            #cos_sim=cos_sim2Str(str(stringa1),str(stringa2))
            #print(cos_sim)
            #cos_sim_list.append(cos_sim)
            if cos_sim<min_cos_sim_match:
                No_match_with_cos_too_small=No_match_with_cos_too_small+1
            
            sim_vector=sim_function(tableA_el,tableB_el) # Modificato
            
            result_list_NOmatch.append((tableA_el,tableB_el,sim_vector, 0))

  
    print(max(len(result_list_match),len(result_list_NOmatch)))
    print(len(result_list_match))
    print(len(result_list_NOmatch))
    
    print("match_tuple: "+str(len(result_list_match)))
    print("no match_tuple: "+str(len(result_list_NOmatch)))
    print("No_match_with_cos_too_small: "+str(No_match_with_cos_too_small))

''' ############# Dataset splitting ####################'''
''' ############# Dataset splitting ####################'''
''' ############# Dataset splitting ####################'''
''' ############# Dataset splitting ####################'''

# Ti restituisce i primi n valori di datatasetOriginal
# con n = len(datasetOriginal) * percentuale
def splitting_dataSet(percentuale, dataSetOriginal):
    lunghezza=int(len(dataSetOriginal)*percentuale)
    "Return first n items of the iterable as a list"
    output=list(islice(dataSetOriginal, lunghezza))
     
    #print("Split length list: ", percentuale) 
    #print("List after splitting", output)
    return output


'''caso label 0/1'''
def splitDataSet01WithPercent(percent, dataSetOriginal, percent1, percent0):
    print(dataSetOriginal)
    lunghezza=int(round(len(dataSetOriginal)*percent))
    percentuale1=int(lunghezza*percent1)
    #print(percentuale1)
    percentuale0=int(lunghezza*percent0)
    #print(percentuale0)
    if int(percentuale1+percentuale0)!=lunghezza:
        percentuale1=percentuale1+1
    output=[]
    i=0
    print('percentuale 1= '+str(percentuale1))
    print('percentuale 0= '+str(percentuale0))
    
    while i<lunghezza:

        for elem in dataSetOriginal:
            if int(elem[2])==1 and percentuale1!=0:
                output.append(elem)
                percentuale1=percentuale1-1
                i=i+1
            elif int(elem[2])==0 and percentuale0!=0:
                    output.append(elem)
                    percentuale0=percentuale0-1
                    i=i+1

    #print(output)
    return output # Modificato


'''caso vector similarity'''
def splitDataSetSIMWithPercent(percent, dataSetOriginal, percent1, percent0):
    lunghezza=int(round(len(dataSetOriginal)*percent))
    percentuale1=int(lunghezza*percent1)
    #print(percentuale1)
    percentuale0=int(lunghezza*percent0)
    #print(percentuale0)
    if int(percentuale1+percentuale0)!=lunghezza:
        percentuale1=percentuale1+1
    output=[]
    i=0
    print('percentuale 1= '+str(percentuale1))
    print('percentuale 0= '+str(percentuale0))
    while i<lunghezza:
        for elem in dataSetOriginal:
            if int(elem[3])==1 and percentuale1!=0:
                output.append(elem)
                percentuale1=percentuale1-1
                i=i+1
            elif int(elem[3])==0 and percentuale0!=0:
                    output.append(elem)
                    percentuale0=percentuale0-1
                    i=i+1
                    
    #print(output)
    return output # Modificato



def buildDataset(dataset_dir,ground_truth,tableL, tableR):
    table1 = pd.read_csv(os.path.join(dataset_dir,tableL))
    table2 = pd.read_csv(os.path.join(dataset_dir,tableR))
    matches_df = pd.read_csv(os.path.join(dataset_dir,ground_truth))

    result_list_match = []
    result_list_NOmatch= []
    matches_ids = list(map(lambda x, y:(x,y), matches_df.id1.values, matches_df.id2.values))
    cos_sim_list=[]
    '''costruisce lista dei match parsando i file di input'''
    for i in range(len(matches_df)):
        curr_lid = matches_df.iloc[i]['id1']
        curr_rid = matches_df.iloc[i]['id2']
        curr_lrecord = table1.iloc[curr_lid]
        curr_rrecord = table2.iloc[curr_rid]
        curr_ltokens = list(map(lambda t:str(t),curr_lrecord.values[1:]))
        curr_rtokens = list(map(lambda t:str(t),curr_rrecord.values[1:]))
        stringa1 = " ".join(curr_ltokens)
        stringa2 = " ".join(curr_rtokens)
        #calcola la cos similarita della tupla i-esima
        cos_sim=get_cosine(text_to_vector(stringa1),text_to_vector(stringa2))
        cos_sim_list.append(cos_sim)
        result_list_match.append((curr_ltokens,curr_rtokens,1))
        #min_cos_sim_match= valore minimo della cos_similarity di tutte quelle in match
    
    min_cos_sim_match=min(cos_sim_list)
    '''costruisce la lista dei NO_match calcolando un min cos similarity'''   
    i=0
    while i<len(result_list_match):

        x = random.randint(0,len(table1)-1)
        y =  random.randint(0,len(table2)-1)            
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
            i += 1

    '''unisce le due liste dei match e No_match alternandole'''
    result_list=[]
    for i in range(max(len(result_list_match),len(result_list_NOmatch))):
        result_list.append(result_list_match[i])
        result_list.append(result_list_NOmatch[i])


    return result_list
