import os
from deeper.csv2dataset import csv_2_dataset_alternate
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
    for t1, t2, lb in data:
        cutPairs.append((t1[:1000],t2[:1000],lb))
    return cutPairs
    
    
# Caricamento dati e split in train, validation e test
def process_data(DATASET_DIR,DATASET_NAME,ground_truth,table1,table2,indici,LOAD_FROM_DISK_DATASET=False,aligned=True):
    if LOAD_FROM_DISK_DATASET:
        
        # Carica dataset salvato su disco.
        allPairs = __load_list(os.path.join(DATASET_DIR,DATASET_NAME+'.pkl'))
        match_number=sum(map(lambda x : x[3] == 1, allPairs))
        print("match_number: " + str(match_number))
        print("len all dataset: "+ str(len(allPairs)))

    else:
        # Necessario inserire le tabelle nell'ordine corrispondente alle coppie della ground truth.

        # Crea il dataset.
        allPairs = csv_2_dataset_alternate(DATASET_DIR,ground_truth=ground_truth,
                                    tableL=table1,tableR=table2,indici=indici,aligned=aligned)
        #per i dataset di Anhai
        #data=parsing_anhai_data(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
        
        # Salva dataset su disco.
        __save_list(allPairs,DATASET_DIR+'data_processed.pkl')

        
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
def process_data_aligned(DATASET_DIR,DATASET_NAME,ground_truth,table1,table2,LOAD_FROM_DISK_DATASET=False):
    if LOAD_FROM_DISK_DATASET:
        
        # Carica dataset salvato su disco.
        allPairs = __load_list(os.path.join(DATASET_DIR,DATASET_NAME+'.pkl'))
        match_number=sum(map(lambda x : x[3] == 1, allPairs))
        print("match_number: " + str(match_number))
        print("len all dataset: "+ str(len(allPairs)))

    else:
        # Necessario inserire le tabelle nell'ordine corrispondente alle coppie della ground truth.

        # Crea il dataset.
        allPairs = csv_2_dataset_alternate_aligned(DATASET_DIR,ground_truth=ground_truth,
                                    tableL=table1,tableR=table2)
        #per i dataset di Anhai
        #data=parsing_anhai_data(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
        
        # Salva dataset su disco.
        __save_list(allPairs,DATASET_DIR+'data_processed.pkl')

        
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
    print('* Preparazione input......', end='', flush=True)
    table1, table2, labels = [], [], []
    for t1, t2, label in data:
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
    print(f'Fatto. {len(labels)} tuple totali, esempio label: {data[0][2]} -> {labels[0]}, Table1 shape: {table1.shape}, Table2 shape: {table2.shape}')

    return table1, table2, labels
