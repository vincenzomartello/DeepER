import os
from deeper.csv2dataset import csv_2_dataset_aligned,csv_2_dataset
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


# Taglia attributi se troppo lunghi
    # Alcuni dataset hanno attributi con descrizioni molto lunghe.
    # Questo filtro limita il numero di caratteri di un attributo a 1000.
def shrink_data(data):
    cutPairs = []
    for t1, t2,label,pairid in data:
        cutPairs.append((t1[:1000],t2[:1000],label,pairid))
    return cutPairs
    
    
# Caricamento dati e split in train, validation e test
def process_data(dataset_dir,dataset_name,ground_truth,table1,table2,indici,load_from_disk_dataset=False):
    if load_from_disk_dataset:
        
        # Carica dataset salvato su disco.
        allPairs = __load_list(os.path.join(dataset_dir,dataset_name+'.pkl'))
        match_number=sum(map(lambda x : x[2] == 1, allPairs))
        print("match_number: " + str(match_number))
        print("len all dataset: "+ str(len(allPairs)))

    else:
        # Necessario inserire le tabelle nell'ordine corrispondente alle coppie della ground truth.

        # Crea il dataset.
        allPairs = csv_2_dataset(dataset_dir,ground_truth=ground_truth,
                                    tableL=table1,tableR=table2,indici=indici)
        #per i dataset di Anhai
        #data=parsing_anhai_data(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
        
        # Salva dataset su disco.
        __save_list(allPairs,DATASET_DIR+DATASET_NAME+'.pkl')

        
    # Dataset per DeepER classico: [(tupla1, tupla2, label), ...].
    deeper_data = shrink_data(allPairs)

    # Split in training set e test set.
    def split_training_test(data, SPLIT_FACTOR = 0.8):
        # Per dividere in maniera random
        np.random.seed(0)
        np.random.shuffle(data)    
        bound = int(len(data) * SPLIT_FACTOR)
        train = data[:bound]
        test = data[bound:]
        
        return train, test


    # Tutti i successivi addestramenti partiranno dal 100% di deeper_train (80% di tutti i dati).
    # Le tuple in deeper_test non verranno mai usate per addestrare ma solo per testare i modelli.
    deeper_train, deeper_test = split_training_test(deeper_data)
    return deeper_train,deeper_test
    
    
# Caricamento dati e split in train, validation e test
def process_data_aligned(dataset_dir,dataset_name,ground_truth,table1,table2,load_from_disk_dataset=False):
    if load_from_disk_dataset:
        
        # Carica dataset salvato su disco.
        allPairs = __load_list(os.path.join(dataset_dir,dataset_name+'.pkl'))
        match_number=sum(map(lambda x : x[2] == 1, allPairs))
        print("match_number: " + str(match_number))
        print("len all dataset: "+ str(len(allPairs)))

    else:
        # Necessario inserire le tabelle nell'ordine corrispondente alle coppie della ground truth.

        # Crea il dataset.
        allPairs = csv_2_dataset_aligned(dataset_dir,ground_truth=ground_truth,
                                    tableL=table1,tableR=table2)
        #per i dataset di Anhai
        #data=parsing_anhai_data(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
        
        # Salva dataset su disco.
        __save_list(allPairs,dataset_dir+dataset_name+'.pkl')

        
    # Dataset per DeepER classico: [(tupla1, tupla2, label), ...].
    deeper_data = shrink_data(allPairs)

    # Split in training set e test set.
    def split_training_test(data, SPLIT_FACTOR = 0.8):
        # Per dividere in maniera random
        np.random.seed(0)
        np.random.shuffle(data)    
        bound = int(len(data) * SPLIT_FACTOR)
        train = data[:bound]
        test = data[bound:]
        
        return train, test


    # Tutti i successivi addestramenti partiranno dal 100% di deeper_train (80% di tutti i dati).
    # Le tuple in deeper_test non verranno mai usate per addestrare ma solo per testare i modelli.
    deeper_train, deeper_test = split_training_test(deeper_data)
    return deeper_train,deeper_test


def __save_list(lista,l_name):
    with open(l_name, 'wb') as f:
        pickle.dump(lista, f)


def __load_list(list_file):
    with open(list_file, 'rb') as f:
        return pickle.load(f)
    
    
    
# InPut: data = [(t1, t2, label),...], tokenizer = tokenizzatore per le tuple
# OutPut: table1, table2 = matrici di tokens, labels matrice con etichette
def data2Inputs(data, tokenizer, categorical=True):
    
    # Limita la sequenza massima di tokens 
    #SEQUENCES_MAXLEN = 500

    # Tokenizza le tuple e prepara l'input per il modello
    table1, table2, labels = [], [], []
    for t1, t2, label,_ in data:
        # Sperimentale: ordino gli attributi per lunghezza decrescente
        # Attributi con molti tokens conengono più informazioni utili 
        #t1 = sorted(t1, key=lambda s: len(s), reverse=True)
        #t2 = sorted(t2, key=lambda s: len(s), reverse=True)                  
        table1.append(' '.join(t1).replace(', ', ' '))
        table2.append(' '.join(t2).replace(', ', ' '))
        labels.append(label)
    table1 = tokenizer.texts_to_sequences(table1)                      
    #table1 = pad_sequences(table1, maxlen=SEQUENCES_MAXLEN, padding='post')
    table1 = pad_sequences(table1, padding='post')
    table2 = tokenizer.texts_to_sequences(table2)        
    #table2 = pad_sequences(table2, maxlen=SEQUENCES_MAXLEN, padding='post')
    table2 = pad_sequences(table2, padding='post')
    if categorical:
        labels = to_categorical(labels)
    else:
        labels = np.array(labels)

    return table1, table2, labels


def data2Inputs_unlabeled(data, tokenizer, categorical=True):
    
    # Limita la sequenza massima di tokens 
    #SEQUENCES_MAXLEN = 500

    # Tokenizza le tuple e prepara l'input per il modello
    table1, table2 = [], []
    for t1,t2 in data:
        # Sperimentale: ordino gli attributi per lunghezza decrescente
        # Attributi con molti tokens conengono più informazioni utili 
        #t1 = sorted(t1, key=lambda s: len(s), reverse=True)
        #t2 = sorted(t2, key=lambda s: len(s), reverse=True)                  
        table1.append(' '.join(t1).replace(', ', ' '))
        table2.append(' '.join(t2).replace(', ', ' '))
    table1 = tokenizer.texts_to_sequences(table1)                      
    #table1 = pad_sequences(table1, maxlen=SEQUENCES_MAXLEN, padding='post')
    table1 = pad_sequences(table1, padding='post')
    table2 = tokenizer.texts_to_sequences(table2)        
    #table2 = pad_sequences(table2, maxlen=SEQUENCES_MAXLEN, padding='post')
    table2 = pad_sequences(table2, padding='post')
    return table1, table2