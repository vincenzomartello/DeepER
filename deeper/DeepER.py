import numpy as np
import tensorflow as tf
from keras.layers import Input, Embedding, LSTM, concatenate, subtract, Dense, Bidirectional, Lambda
from keras.models import Model, load_model
from keras.initializers import Constant
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model, to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from deeper.data import data2Inputs
import os


# Workaround per warnings su codice deprecato
#tf.logging.set_verbosity(tf.logging.ERROR)


# InPut: Nome del file con gli embeddings
# Output: Un dizionario con tutti gli embeddings: {parola: embedding}
def init_embeddings_index(embeddings_file):

    print('* Costruzione indice degli embeddings.....', end='', flush=True)
    embeddings_index = {}
    with open(embeddings_file, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    print(f'Fatto. {len(embeddings_index)} embeddings totali.')
    
    return embeddings_index


# InPut: Nome del file contenente gli embeddings
# OutPut: Un modello che converte vettori di token in vettori di embeddings ed un tokenizzatore
def init_embeddings_model(embeddings_index):

    print('* Creazione del modello per il calcolo degli embeddings....', flush=True)

    print('* Inizializzo il tokenizzatore.....', end='', flush=True)
    tokenizer = Tokenizer(filters='')
    tokenizer.fit_on_texts(embeddings_index.keys())
    words_index = tokenizer.word_index
    print(f'Fatto: {len(words_index)} parole totali.')    

    print('* Preparazione della matrice di embedding.....', end='', flush=True)
    embedding_dim = len(embeddings_index['cat']) # :3  
    num_words = len(words_index) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in words_index.items():    
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Le parole di cui non viene trovato l'embedding avranno vettore nullo
            embedding_matrix[i] = embedding_vector
    print(f'Fatto. Dimensioni matrice embeddings: {embedding_matrix.shape}')

    print('\n°°° EMBEDDING MODEL °°°')                       
    # Input layer: due tuple, ciascuna tupla è 
    # una sequenza di token (numeri)
    input_a = Input(shape=(None,), name='Tupla_A')
    input_b = Input(shape=(None,), name='Tupla_B')

    # Embedding lookup layer (shared)
    # num_words = numero di parole uniche usate in totale, 
    # embedding_dim = dimensione vettore embedding
    embedding = Embedding(num_words,
                          embedding_dim,
                          embeddings_initializer=Constant(embedding_matrix),
                          trainable=False, 
                          name='Embedding_lookup')
    embedding_a = embedding(input_a)
    embedding_b = embedding(input_b)

    # Creazione modello per gli embeddings
    embeddings_model = Model(inputs=[input_a, input_b], outputs=[embedding_a, embedding_b])
    embeddings_model.summary() 
    
    return embeddings_model, tokenizer


# OutPut: Il modello DeepER compilato pronto per l'addestramento
def init_DeepER_model(embedding_dim):

    print('\n°°° DeepER Model °°°')
    # Input layer: due sequenze di embeddings
    emb_a = Input(shape=(None, embedding_dim), name='Embeddings_seq_a')
    emb_b = Input(shape=(None, embedding_dim), name='Embeddings_seq_b')

    # Composition layer
    # Bidirectional
    lstm = Bidirectional(LSTM(150), name='Composition')
    # Unidirectional
    #lstm = LSTM(100, name='Composition')
    lstm_a = lstm(emb_a)
    lstm_b = lstm(emb_b)

    # Similarity layer (subtract or hadamart prod),
    # vedi Keras core layers e Keras merge layers per altre tipologie
    # Subtract
    #similarity = subtract([lstm_a, lstm_b], name='Similarity')
    # Hadamart    
    similarity = Lambda(lambda ts: ts[0] * ts[1], name='Similarity')([lstm_a, lstm_b])

    # Dense layer
    dense = Dense(300, activation='relu', name='Dense1')(similarity)
    dense = Dense(300, activation='relu', name='Dense2')(dense)
    
    # Classification layer
    output = Dense(2, activation='softmax', name='Classification')(dense)

    # Creazione modello
    deeper_model = Model(inputs=[emb_a, emb_b], outputs=[output])

    # Compilazione per addestramento
    deeper_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #deeper_model.summary()

    return deeper_model



# InPut: modello, nuovo output layer
# OutPut: nuovo modello con l'output layer specificato al posto di quello precedente
# I layers dovrebbero essere condivisi con il modello passato per parametro
def replace_last_layer(model, new_layer):

    x = model.layers[-2].output
    x = new_layer(x)

    return Model(inputs=model.input, outputs=x)


# InPut: Una lista di triple [(tup1, tup2, label), ...], il modello da addestrare...
# Output: Il modello addestrato
#def train_DeepER(deeper_model, data, embeddings_index):  
def train_model_ER(data, model,embeddings_model, tokenizer,best_save_path='models',pretraining=False, metric='val_accuracy', end=''):

    if pretraining:
        model_name = 'VinSim'
        table1, table2, labels = data2Inputs(data, tokenizer, categorical=False)
    else:
        model_name = 'DeepER'
        table1, table2, labels = data2Inputs(data, tokenizer)         
    
    # Preparazione embeddings (tokens -> embeddings)
    x1, x2 = embeddings_model.predict([table1, table2])        
    
    # Early stopping (arresta l'apprendimento se non ci sono miglioramenti)
    es = EarlyStopping(monitor=metric, min_delta=0, verbose=1, patience=7)
    # Model checkpointing (salva il miglior modello fin'ora)
    mc = ModelCheckpoint(
        os.path.join(best_save_path,f'{model_name}_best_model{end}.h5'), monitor=metric, verbose=1,save_best_only=True)
    # Addestramento modello
    param_batch_size = round(len(data) * 0.015) + 1
    print('Batch size:', param_batch_size)
    model.fit([x1, x2], labels, batch_size=param_batch_size, epochs=64, validation_split=0.2, callbacks=[es, mc])

    # Carica il miglior modello checkpointed
    model = load_model(os.path.join(best_save_path,f'{model_name}_best_model{end}.h5'))   

    return model

def train_model_ROUND_ER(data, model, embeddings_model, tokenizer, metric='val_accuracy', end=''):

    model_name = 'VinSim'

    table1, table2, labels = data2Inputs(data, tokenizer)         
    
    # Preparazione embeddings (tokens -> embeddings)
    x1, x2 = embeddings_model.predict([table1, table2])        
    
    # Early stopping (arresta l'apprendimento se non ci sono miglioramenti)
    es = EarlyStopping(monitor=metric, min_delta=0, verbose=1, patience=7)
    # Model checkpointing (salva il miglior modello fin'ora)
    mc = ModelCheckpoint(f'{model_name}_best_model{end}.h5', monitor=metric, verbose=1, save_best_only=True)
    # Addestramento modello
    param_batch_size = round(len(data) * 0.015) + 1
    print('Batch size:', param_batch_size)
    model.fit([x1, x2], labels, batch_size=param_batch_size, epochs=64, validation_split=0.2, callbacks=[es, mc])

    # Carica il miglior modello checkpointed
    model = load_model(f'{model_name}_best_model{end}.h5')   

    return model


# F-Measure
# InPut: Dati nel formato [(tupla1, tupla2, label),...], un modello da testare
# OutPut: Statistiche sul modello 
def model_statistics(data, model, embeddings_model, tokenizer):

    print('* Avvio test metriche....', flush=True)

    corpus_size = len(data)

    no_match_count = 0
    match_count = 0    
    for tri in data:
        if tri[2] == 1:
            match_count += 1
        else:
            no_match_count += 1

    print(f'-- Corpus size: {corpus_size}')
    print(f'-- Non Match: {no_match_count}')
    print(f'-- Match: {match_count}')
  
    # Crea matrici di tokens e labels
    table1, table2, _ = data2Inputs(data, tokenizer)

    # Calcola inputs di embeddings
    emb1, emb2 = embeddings_model.predict([table1, table2])

    match_retrieved = [] 
    
    true_match = 0
    
    print('* Evaluating: ', end='', flush=True)
    
    pred_matrix = model.predict([emb1, emb2])

    for i in range(corpus_size):
        prediction = pred_matrix[i]        
        if np.argmax(prediction) == 1:
            match_retrieved.append(data[i])            
            # Conta predizioni di match corrette
            if data[i][2] == 1:
                true_match += 1
        if i % (corpus_size // 10) == 0:
            print(f'=', end='', flush=True)

    print('|')
    
    # Per gestire eccezioni in cui le prestazioni sono talmente negative
    # da non recuperare alcun valore inizializza l'fmeasure a -1
    fmeasure = -1
    try:
        retrieved_size = len(match_retrieved)
        precision = true_match / retrieved_size
        recall =  true_match /  match_count
        fmeasure = 2 * (precision * recall) / (precision + recall)

        print(f'Precision: {precision}, Recall: {recall}, f1-score: {fmeasure}')
        print(f'Total retrieved: {retrieved_size}, retrieved/total matches: {true_match}/{match_count}')

    except:

        print(f'Error. Retrieved = {retrieved_size}, Matches = {match_count} ')

    return fmeasure



# TEST AREA #
if __name__ == "__main__":

    #ground_truth='matches_dblp_scholar.csv'
    #table1='dblp.csv'
    #table2='google_scholar.csv'

    #ground_truth='DBLP-Scholar_perfectMapping.csv'
    #table1='DBLP1.csv'
    #table2='Scholar.csv'

    #ground_truth='Amzon_GoogleProducts_perfectMapping.csv'
    #table1='Amazon.csv'
    #table2='GoogleProducts.csv'

    ground_truth='matches_walmart_amazon.csv'
    table1='walmart.csv'
    table2='amazon.csv'

    GROUND_TRUTH_FILE = ground_truth
    TABLE1_FILE = table1
    TABLE2_FILE = table2

    # Carica da csv le tuple
    print('* Loading dataset.....', end='', flush=True)
       
    from csv2dataset import csv_2_dataset
    from generate_similarity_vector import generate_similarity_vector
    
    index_wa = [(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]
    
    # Funzione ausiliaria per la similarità
    simf = lambda a, b: generate_similarity_vector(a, b, flag='W-A')
    
    # Crea il dataset
    data = csv_2_dataset(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, index_wa, sim_function=simf)
    
    # Ricrea il dataset senza sim vector per addestramento
    data = list(map(lambda q: (q[0], q[1], q[3]), data))
    print(f'Fatto. {len(data)} coppie di tuple.')    

    # Caricamento strutture dati e modelli ausiliari
    embeddings_index = init_embeddings_index('glove.6B.50d.txt')
    emb_dim = len(embeddings_index['cat']) # :3
    embeddings_model, tokenizer = init_embeddings_model(embeddings_index)
    
    # Addestramento (o caricamento)
    #deeper_model = load_model('DeepER_best_model_75_pre_wa.h5')    
    split = int(len(data) * 0.8)
    deeper_model = train_model_ER(data[:split], 
                                  init_DeepER_model(emb_dim), 
                                  embeddings_model, 
                                  tokenizer, 
                                  pretraining=False, 
                                  end='_test')    

    '''
    # Test integrità dei pesi, attacca e stacca il layer di output e riavvia l'addestramento 
    from VinSim import replace_last_layer
    # Sostituisce il layer finale con quello di similarità
    vin_model = replace_last_layer(deeper_model, Dense(8, activation='sigmoid', name='VinSim'))
    vin_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #vin_model.summary()
    # segue addestramento opportuno con vin_model.fit(.....)

    # Ri-applica il layer originale per classificazione
    deeper_model = replace_last_layer(vin_model, Dense(2, activation='softmax', name='Classification'))
    deeper_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #deeper_model.summary()
    # restart training 
    deeper_model = train_DeepER(deeper_model, data, emb_index)
    '''
    # Statistiche su piccolo dataset standard bilanciato
    model_statistics(data[split:], deeper_model, embeddings_model, tokenizer)
    
    '''
    # Statistiche su dataset cattivo (solo 3 tuple in match)
    bad_boy = []
    good_count = 3
    for tripla in data:
        if tripla[2] == 0:
            bad_boy.append(tripla)
        elif good_count > 0:
            bad_boy.append(tripla)
            good_count -= 1
        else:
            continue

    model_statistics(bad_boy, deeper_model, embeddings_model, tokenizer)
    '''

