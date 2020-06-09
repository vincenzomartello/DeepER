import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


class Dataset:


    def __init__(self,tokenizer,embedding_model,data,categorical=True,unlabeled=False):
        self.tokenizer = tokenizer
        self.embedding_model = embedding_model
        self.data = {}
        if unlabeled:
            self._data2Inputs_unlabeled(data,categorical)
        else:
            self._data2Inputs(data,categorical)



    # OutPut: left_records, right_records = matrici di tokens, labels matrice con etichette
    def _data2Inputs(self,data,categorical):

        # Limita la sequenza massima di tokens
        #SEQUENCES_MAXLEN = 500
        # Tokenizza le tuple e prepara l'input per il modello
        left_records, right_records, labels = [], [], []
        for t1, t2, label,_ in data:
            #t1 = sorted(t1, key=lambda s: len(s), reverse=True)
            #t2 = sorted(t2, key=lambda s: len(s), reverse=True)
            left_records.append(' '.join(t1).replace(', ', ' '))
            right_records.append(' '.join(t2).replace(', ', ' '))
            labels.append(label)
        left_records = self.tokenizer.texts_to_sequences(left_records)
        #left_records = pad_sequences(left_records, maxlen=SEQUENCES_MAXLEN, padding='post')
        left_records = pad_sequences(left_records, padding='post')
        right_records = self.tokenizer.texts_to_sequences(right_records)
        #right_records = pad_sequences(right_records, maxlen=SEQUENCES_MAXLEN, padding='post')
        right_records = pad_sequences(right_records, padding='post')
        if categorical:
            labels = to_categorical(labels)
        else:
            labels = np.array(labels)
            
        left_embeddings,right_embeddings = self.embedding_model.predict([left_records,right_records])
        self.data['left_embeddings'] = left_embeddings
        self.data['right_embeddings'] = right_embeddings
        self.data['labels'] = labels
        
        return
    
    
    def _data2Inputs_unlabeled(self,data,categorical):
    
        # Limita la sequenza massima di tokens 
        #SEQUENCES_MAXLEN = 500

        # Tokenizza le tuple e prepara l'input per il modello
        left_records, right_records = [], []
        for t1,t2 in data:                
            left_records.append(' '.join(t1).replace(', ', ' '))
            right_records.append(' '.join(t2).replace(', ', ' '))
        left_records = self.tokenizer.texts_to_sequences(left_records)                      
        #table1 = pad_sequences(table1, maxlen=SEQUENCES_MAXLEN, padding='post')
        left_records = pad_sequences(left_records, padding='post')
        right_records = self.tokenizer.texts_to_sequences(right_records)        
        #table2 = pad_sequences(table2, maxlen=SEQUENCES_MAXLEN, padding='post')
        right_records = pad_sequences(right_records, padding='post')
        
        left_embeddings,right_embeddings = self.embedding_model.predict([left_records,right_records])
        self.data['left_embeddings'] = left_embeddings
        self.data['right_embeddings'] = right_embeddings
        return
    
    
