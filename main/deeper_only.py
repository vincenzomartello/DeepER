# -*- coding: utf-8 -*-
"""
training Deeper
"""

from DeepER import init_embeddings_index, init_embeddings_model, init_DeepER_model, train_model_ER, replace_last_layer, model_statistics
from csv2dataset import parsing_anhai_data, splitting_dataSet, csv_2_datasetALTERNATE, csvTable2datasetRANDOM
from keras.models import load_model
from keras.layers import Dense
from plotly import graph_objs as go
import plotly.offline as pyo
from random import shuffle
import utilist as uls

# Imposta manualmente a True per caricare da disco tutti i modelli salvati e i dataset già parsati. 
# Imposta manualmente a False per ri-eseguire tutti gli addestramenti e ricostruire il dataset.
LOAD_FROM_DISK_DATASET=False
LOAD_FROM_DISK_MODEL = False
# Il nome con cui saranno etichettati i files prodotti
DATASET_NAME = 'wa_Anhai'# Esempio: 'WA'

# Caricamento strutture dati e modelli ausiliari.
embeddings_index = init_embeddings_index('glove.6B.50d.txt')
emb_dim = len(embeddings_index['cat']) # :3
embeddings_model, tokenizer = init_embeddings_model(embeddings_index)

# Caricamento dati e split iniziale.
if LOAD_FROM_DISK_DATASET:
    
    # Carica dataset salvato su disco.
    data = uls.load_list(f'dataset_{DATASET_NAME}')
    match_number=sum(map(lambda x : x[3] == 1, data))
    print(match_number)
    print(len(data))

else:
    
    GROUND_TRUTH_FILE = 'matches_fodors_zagats.csv'# Esempio: 'matches_walmart_amazon.csv'
    # Necessario inserire le tabelle nell'ordine corrispondente alle coppie della ground truth.
    TABLE1_FILE = 'fodors.csv'# Esempio: 'walmart.csv'
    TABLE2_FILE = 'zagats.csv'# Esempio: 'amazon.csv'

    # Coppie di attributi considerati allineati.
    att_indexes = [(1, 1), (2, 2), (3, 3), (4, 4),(5, 5), (6, 6)]# Esempio: [(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]


    # Similarity callbacks
    #simf= lambda a,b: sim_hamming(a,b)


    # Crea il dataset.
    data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
    
    #per i dataset di Anhai
    #data=parsing_anhai_data(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes)
    # Salva dataset su disco.
    uls.save_list(data, f'dataset_{DATASET_NAME}')

    
# Dataset per DeepER classico: [(tupla1, tupla2, label), ...].
deeper_data = list(map(lambda q: (q[0], q[1], q[3]), data))


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

deeper_data = shrink_data(deeper_data)


# Split in training set e test set.
def split_training_test(data, SPLIT_FACTOR = 0.8):     
    bound = int(len(data) * SPLIT_FACTOR)
    train = data[:bound]
    test = data[bound:]
    
    return train, test


# Tutti i successivi addestramenti partiranno dal 100% di deeper_train (80% di tutti i dati).
# Le tuple in deeper_test non verranno mai usate per addestrare ma solo per testare i modelli.
deeper_train, deeper_test = split_training_test(deeper_data)


# InPut: Percentuale di dati considerata per l'addestramento. 
# OutPut: DeepER addestrato sul taglio scelto.
def get_DeepER(perc):
   
    sub_data = splitting_dataSet(perc, deeper_train)    
    
    if LOAD_FROM_DISK_MODEL:
        
        # Carica da disco.
        print(f'Loading DeepER_best_model_{int(perc*100)}_{DATASET_NAME}.h5', end='', flush=True)
        deeper_model = load_model(f'DeepER_best_model_{int(perc*100)}_{DATASET_NAME}.h5')
        print('  ->  Done')        
                
    else:
        
        # Inizializza il modello.
        deeper_model = init_DeepER_model(emb_dim)
        deeper_model.summary()
        # Avvio addestramento.
        deeper_model = train_model_ER(sub_data, 
                                      deeper_model, 
                                      embeddings_model, 
                                      tokenizer, 
                                      pretraining=False, 
                                      end=f'_{int(perc*100)}_{DATASET_NAME}')
        
    return deeper_model


# Avvio addestramenti o carica da disco.
deeper_model_5 = get_DeepER(0.05)
deeper_model_10 = get_DeepER(0.1)
deeper_model_25 = get_DeepER(0.25)
deeper_model_50 = get_DeepER(0.5)
deeper_model_75 = get_DeepER(0.75)
deeper_model_100 = get_DeepER(1)

# Misurazione dell'f-measure sullo stesso test set con i diversi modelli.
fm_model_standard = [model_statistics(deeper_test, deeper_model_100, embeddings_model, tokenizer),
                     model_statistics(deeper_test, deeper_model_75, embeddings_model, tokenizer),
                     model_statistics(deeper_test, deeper_model_50, embeddings_model, tokenizer),
                     model_statistics(deeper_test, deeper_model_25, embeddings_model, tokenizer),
                     model_statistics(deeper_test, deeper_model_10, embeddings_model, tokenizer),
                     model_statistics(deeper_test, deeper_model_5, embeddings_model, tokenizer)]

print(fm_model_standard)