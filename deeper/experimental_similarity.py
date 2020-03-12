# pip install strsim
from similarity.levenshtein import Levenshtein
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.damerau import Damerau
from similarity.optimal_string_alignment import OptimalStringAlignment
from similarity.jarowinkler import JaroWinkler
from similarity.longest_common_subsequence import LongestCommonSubsequence
from similarity.metric_lcs import MetricLCS
from similarity.ngram import NGram
from similarity.qgram import QGram
from similarity.cosine import Cosine
from similarity.jaccard import Jaccard
from similarity.sorensen_dice import SorensenDice
from scipy.spatial.distance import euclidean, cosine, cityblock
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np

# Inizializza all'import
levenshtein = Levenshtein()
norm_levenshtein = NormalizedLevenshtein()
damerau = Damerau()
optimal_string_alignment = OptimalStringAlignment()
jarowinkler = JaroWinkler()
lcs = LongestCommonSubsequence()
metric_lcs = MetricLCS()
ngram = NGram()
qgram = QGram()
dice = SorensenDice()
cos = Cosine(5)
jaccard = Jaccard(5)

similarity_functions = [norm_levenshtein.similarity,
                        lambda a, b: 1 - metric_lcs.distance(a, b),
                        lambda a, b: 1 - ngram.distance(a, b),
                        cos.similarity,
                        dice.similarity]

def mono_vector0(tup1, tup2):

    str1 = ' '.join(tup1).lower()
    str2 = ' '.join(tup2).lower()

    simv = list(map(lambda x: x(str1, str2), similarity_functions))
    
    
    
    return simv

# InPut: Due tuple
# OutPut: Un vettore monodimensionale con la media di alcune funzioni di similarità
def mono_vector(tup1, tup2):

    str1 = ' '.join(tup1).lower()
    str2 = ' '.join(tup2).lower()

    simv = list(map(lambda x: x(str1, str2), similarity_functions))
    
    return [np.mean(simv)]

sim_fun4 = [norm_levenshtein.similarity,
             cos.similarity,
             lambda a, b: 1 - metric_lcs.distance(a, b),
             lambda a, b: 1 - ngram.distance(a, b)]


# InPut: Due tuple
# OutPut: Un vettore monodimensionale con la media di alcune funzioni di similarità
#ex mono_vector
def mono_sim_fun4(tup1, tup2):

    str1 = ' '.join(tup1).lower()
    str2 = ' '.join(tup2).lower()

    simv = list(map(lambda x: x(str1, str2), sim_fun4))
    
    return [np.mean(simv)]

sim_fun2 = [norm_levenshtein.similarity,
             lambda a, b: 1 - metric_lcs.distance(a, b),
             dice.similarity,
             lambda a, b: 1 - ngram.distance(a, b)]



# InPut: Due tuple
# OutPut: Un vettore monodimensionale con la media di alcune funzioni di similarità
#ex mono_vector
def mono_sim_fun2(tup1, tup2):

    str1 = ' '.join(tup1).lower()
    str2 = ' '.join(tup2).lower()

    simv = list(map(lambda x: x(str1, str2), sim_fun2))
    
    return [np.mean(simv)]


import textdistance as txd
sfs = [txd.damerau_levenshtein.normalized_similarity,      
       txd.jaro_winkler.normalized_similarity,
       #txd.hamming.normalized_similarity,
       txd.cosine.normalized_similarity]
# Versione velocizzata
def mono_vector_fast(tup1, tup2):
    str1 = ' '.join(tup1).lower()
    str2 = ' '.join(tup2).lower()

    simv = list(map(lambda x: x(str1, str2), sfs))
    
    return [np.mean(simv)]


# InPut: Una stringa di testo = Un singolo attributo
# OutPut: Il vettore medio degli embeddings associati a ciascuna parola
def mean_emb(s, embeddings_index):

    s = s.lower().replace(',', ' ')
    
    mat = []

    for token in s.split():
        try:
            mat.append(embeddings_index[token])
        except:
            continue

    if len(mat) != 0:
        return np.mean(mat, axis=0, keepdims=True)
    else:
        raise ValueError('Missing tokens')


# InPut: Due tuple con uguale numero di attributi, l'indice degli embeddings {<parola: vettore>}
# Output: Un vettore di dimensione d = numero di attributi delle due tuple
def cosine_similarity_vector(tup1, tup2, emb_index):
    
    assert len(tup1) == len(tup2), 'Le tuple devo avere lo stesso numero di attributi!'
    
    simv = []

    for i in range(len(tup1)):
        try:
            mv1 = mean_emb(tup1[i], emb_index)
        except:
            mv1 = mean_emb(' '.join(tup1), emb_index)
        
        try:
            mv2 = mean_emb(tup2[i], emb_index)
        except:
            mv2 = mean_emb(' '.join(tup2), emb_index)

        simv.append(cosine(mv1, mv2)/2)   
    
    return list(map(lambda x: 1 - x, simv)) #np.ndarray.tolist(1 - normalize([simv], norm='l1')[0])


# InPut: Due tuple con uguale numero di attributi, l'indice degli embeddings {<parola: vettore>}
# Output: Un vettore di dimensione d = numero di attributi delle due tuple
def distance_similarity_vector(tup1, tup2, emb_index, metric='euclidean'):
    
    assert len(tup1) == len(tup2), 'Le tuple devo avere lo stesso numero di attributi!'

    metrics = {'euclidean': euclidean, 'manhattan': cityblock, 'cosine': cosine}
    distance = metrics[metric]
    
    simv = []

    for i in range(len(tup1)):
        try:
            mv1 = mean_emb(tup1[i], emb_index)
        except:
            mv1 = mean_emb(' '.join(tup1), emb_index)
        
        try:
            mv2 = mean_emb(tup2[i], emb_index)
        except:
            mv2 = mean_emb(' '.join(tup2), emb_index)

        simv.append(distance(mv1, mv2))    
    
    #simv = np.ndarray.tolist(normalize([simv], norm='l2')[0])

    return simv



# TEST AREA #     
if __name__ == "__main__":

    from DeepER import init_embeddings_index

    embeddings_index = init_embeddings_index('glove.6B.50d.txt')

    t1 = ['Nikon COOLPIX L120 14.1MP Bronze Digital Camera 4.5-94.5mm Zoom NIKKOR Lens 3.0  LCD Display HD Video w 50 Bonus Prints', 'Digital Cameras', 'Nikon', '26255', '274.36']
    t2 = ['Kodak Easyshare M580 14 MP Digital Camera with 8x Wide Angle Optical Zoom and 3.0-Inch LCD Blue', 'Point Shoot Digital Cameras', 'Kodak', 'EasyShare M580 Blue', ' ']
    
    print('Mono vector:', mono_vector(t1, t2))
    
    print('Mono vector fast:', mono_vector_fast(t1, t2))

    print('Cosine similarity:', cosine_similarity_vector(t1, t2, embeddings_index))

    print('Distance similarity:', distance_similarity_vector(t1, t2, embeddings_index))


    from csv2dataset import csvTable2datasetRANDOM, csv_2_datasetALTERNATE
    
    GROUND_TRUTH_FILE = 'WA/matches_walmart_amazon.csv'
    TABLE1_FILE = 'WA/walmart.csv'
    TABLE2_FILE = 'WA/amazon.csv'

    # Coppie di attributi considerati per Walmart Amazon (Walmart_att, Amazon_att).
    att_indexes = [(5, 9), (4, 5), (3, 3), (14, 4), (6, 11)]
    #simf = lambda a, b: cosine_similarity_vector(a, b, embeddings_index)
    simf = lambda a, b: mono_vector_fast(a, b)
    data = csv_2_datasetALTERNATE(GROUND_TRUTH_FILE, TABLE1_FILE, TABLE2_FILE, att_indexes, simf)
    for i in range(100):
        print(data[i][0])
        print(data[i][1])
        print(data[i][2])
        print(data[i][3])
        if data[i][2][0] >= 0.6:
            print(1)
        else:
            print(0)
        print()
        
    
    
    
   