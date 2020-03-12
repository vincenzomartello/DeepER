import os
from deeper.csv2dataset import csv_2_datasetALTERNATE
import numpy as np
import pickle


# Taglia attributi se troppo lunghi
    # Alcuni dataset hanno attributi con descrizioni molto lunghe.
    # Questo filtro limita il numero di caratteri di un attributo a 1000.
def shrink_data(data):
    def cut_string(s):
        if len(s) >= 1000:
            return s[:1000]
        else:
            return s
        
    temp = []
    for t1, t2, lb in data:
        t1 = list(map(cut_string, t1))
        t2 = list(map(cut_string, t2))
        temp.append((t1, t2, lb))
            
    return temp
    
    
# Caricamento dati e split in train, validation e test
def process_data(DATASET_DIR,DATASET_NAME,ground_truth,table1,table2,LOAD_FROM_DISK_DATASET=False):
    if LOAD_FROM_DISK_DATASET:
        
        # Carica dataset salvato su disco.
        allPairs = __load_list(os.path.join(DATASET_DIR,DATASET_NAME+'.pkl'))
        match_number=sum(map(lambda x : x[3] == 1, allPairs))
        print("match_number: " + str(match_number))
        print("len all dataset: "+ str(len(allPairs)))

    else:
        # Necessario inserire le tabelle nell'ordine corrispondente alle coppie della ground truth.

        # Coppie di attributi considerati allineati.
        att_indexes = [(1, 1), (2, 2), (3, 3), (4, 4),(5, 5), (6, 6)]# Esempio: [(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]


        # Crea il dataset.
        allPairs = csv_2_datasetALTERNATE(DATASET_DIR,ground_truth=ground_truth,
                                    tableL=table1,tableR=table2,indici=att_indexes)
        #per i dataset di Anhai
        #data=parsing_anhai_data(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
        
        # Salva dataset su disco.
        __save_list(allPairs,DATASET_DIR+'data_processed.pkl')

        
    # Dataset per DeepER classico: [(tupla1, tupla2, label), ...].
    deeper_data = list(map(lambda q: (q[0], q[1], q[3]), allPairs))

    deeper_data = shrink_data(deeper_data)

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