from pyjarowinkler import distance
import textdistance

'''  le tuple di input per la funzione che genera il vettore di similarita dovranno essere formattate nel seguente modo, a seconda del dataset che si utilizza per il modello '''

# ------------------------------ ACM - DBLP2 (LIBRI) - uguale a DBPL-Scholar e CITATIONS -----------------------------------------------
#FLAG= 'BOOKS'

# TUPLA=                        [title, authors, venue, year]
# per ACM.csv colonne:          [  1,      2,      3,     4 ]
# per DBLP2.csv colonne:        [  1,      2,      3,     4 ]


# ------------------------------ WALMART - AMAZON (PRODUCTS) -----------------------------------------------
#FLAG= 'W-A'

# TUPLA1=                  [title, groupname, brand, modelno, price]
# per walmart.csv colonne: [  5,      4,        3,     14,      6  ]

# TUPLA2=                  [title, category1, brand, modelno, price]---- pcategory1 in colonna [6] corrisponde di piu alla groupname di walmart
# per amazon.csv colonne:  [  9,       5,       3,      4,     11  ]
   


# ------------------------------ AMAZON - GOOGLE (PRODUCTS) -----------------------------------------------
#FLAG= 'A-G'

# TUPLA1=                          [title, description, manufacturer, price]
# per Amazon.csv colonne:          [  1,        2,           3,         4  ]

# TUPLA2=                          [name, description, manufacturer, price]
# per GoogleProducts.csv colonne:  [ 1,        2,           3,         4  ]    



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



# genera vettore di similarita per dataset in formato BOOKS
def books_vector(tuple1, tuple2):

    jawi = distance.get_jaro_distance(tuple1[0], tuple2[0], winkler=True, scaling=0.1)   # jaro-winkler similarity
    jacc = textdistance.jaccard.normalized_similarity(tuple1[1], tuple2[1])              # jaccard similarity
    cos = textdistance.cosine.normalized_similarity(tuple1[2], tuple2[2])               # cosine similarity

    # calcolo similarità tra anni normalizzata, i valori in input saranno stringhe
    year = 0.0

    try:
        year = numeric_normalized_similarity(float(tuple1[3]),float(tuple2[3]))
    except ValueError as val_error:
        # could not convert string to float
        pass


    concat1 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1], tuple2[0]+' '+tuple2[1])
    concat2 = textdistance.cosine.normalized_similarity(tuple1[2]+' '+str(tuple1[3]), tuple2[2]+' '+str(tuple2[3]))
    concat3 = textdistance.jaccard.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))


#    lev = textdistance.levenshtein.normalized_similarity(tuple1[0], tuple2[0])           # levenshtein similarity
#    sodi = textdistance.sorensen_dice.normalized_similarity(tuple1[0], tuple2[0])        # sorensen-dice similarity
#    w1 = (0.7*jawi) + (0.3*jacc)                                                         # weighed similarity 1
#    w2 = (0.7*lev) + (0.3*sodi)                                                          # weighed similarity 2

    vector = [float('%.3f'%(jawi)), float('%.3f'%(jacc)), float('%.3f'%(cos)), float('%.3f'%(year)), 
                float('%.3f'%(concat1)), float('%.3f'%(concat2)), float('%.3f'%(concat3))]   
    return vector

# genera vettore di similarita per dataset in formato W-A
def wa_vector(tuple1, tuple2):

    jawi1 = distance.get_jaro_distance(tuple1[0], tuple2[0], winkler=True, scaling=0.1)
    cos1 = textdistance.cosine.normalized_similarity(tuple1[1], tuple2[1])
    jawi2 = distance.get_jaro_distance(tuple1[2], tuple2[2], winkler=True, scaling=0.1)
    cos2 = textdistance.cosine.normalized_similarity(tuple1[3], tuple2[3])

    # calcolo similarità tra prezzi normalizzata, i valori in input saranno stringhe
    price = 0.0

    try:
        price = numeric_normalized_similarity(float(tuple1[4]),float(tuple2[4]))
    except ValueError as val_error:
        # could not convert string to float
        pass


    concat1 = textdistance.sorensen_dice.normalized_similarity(tuple1[0]+' '+tuple1[1], tuple2[0]+' '+tuple2[1])
    concat2 = textdistance.cosine.normalized_similarity(tuple1[2]+' '+tuple1[3]+' '+str(tuple1[4]), tuple2[2]+' '+tuple2[3]+' '+str(tuple2[4]))
    concat3 = textdistance.cosine.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+tuple1[3]+' '+str(tuple1[4]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+tuple2[3]+' '+str(tuple2[4]))


    vector = [float('%.3f'%(jawi1)), float('%.3f'%(cos1)), float('%.3f'%(jawi2)), float('%.3f'%(cos2)),
                float('%.3f'%(price)), float('%.3f'%(concat1)), float('%.3f'%(concat2)), float('%.3f'%(concat3))]
    return vector

# genera vettore di similarita per dataset in formato A-G
def ag_vector(tuple1, tuple2):

    jawi0 = distance.get_jaro_distance(tuple1[0], tuple2[0], winkler=True, scaling=0.1)
    cos1 = textdistance.cosine.normalized_similarity(tuple1[1], tuple2[1])
    cos2 = textdistance.cosine.normalized_similarity(tuple1[2], tuple2[2])

    # calcolo similarità tra prezzi normalizzata, i valori in input saranno stringhe
    price = 0.0

    try:
        price = numeric_normalized_similarity(float(tuple1[3]),float(tuple2[3]))
    except ValueError as val_error:
        # could not convert string to float
        pass

    concat1 = distance.get_jaro_distance(tuple1[0]+' '+tuple1[1], tuple2[0]+' '+tuple2[1], winkler=True, scaling=0.1)
    concat2 = textdistance.cosine.normalized_similarity(tuple1[2]+' '+str(tuple1[3]), tuple2[2]+' '+str(tuple2[3]))
    concat3 = textdistance.cosine.normalized_similarity(tuple1[0]+' '+tuple1[1]+' '+tuple1[2]+' '+str(tuple1[3]),
                                                         tuple2[0]+' '+tuple2[1]+' '+tuple2[2]+' '+str(tuple2[3]))


    vector = [float('%.3f'%(jawi0)), float('%.3f'%(cos1)), float('%.3f'%(cos2)), float('%.3f'%(price)), 
                float('%.3f'%(concat1)), float('%.3f'%(concat2)), float('%.3f'%(concat3))]
    return vector



# funzione di default dataset, si basa sul fatto che le due tuple in input abbiano stesso numero di attributi e siano allineati
def default_vector(tuple1, tuple2):
    vector = []
    tconc1 = ' '
    tconc2 = ' '
    for i in range(len(tuple1)):
        cos = float('%.3f'%(textdistance.cosine.normalized_similarity(str(tuple1[i]), str(tuple2[i]))))
        vector.append(cos)
        tconc1 += str(tuple1[i])
        tconc2 += str(tuple2[i])
    
    concat = float('%.3f'%(textdistance.cosine.normalized_similarity(tconc1, tconc2)))
    vector.append(concat)

    return vector


'''funzione per costruire il vettore di similarity 
    input: due tuple di stringhe e il flag che identifica il dataset di input
    output: vettore di similarita'''
def generate_similarity_vector(tuple1, tuple2, flag="default"):

    sim_vector=[]
    tup1=tuple1
    tup2=tuple2

    for i in range(len(tup1)):
        if tup1[i] == "" or tup1[i] is None:
            tup1[i] = " "
    
    for i in range(len(tup2)):
        if tup2[i] == "" or tup2[i] is None:
            tup2[i] = " "

    if (flag == "BOOKS"): # book datasets
        sim_vector = books_vector(tup1, tup2)
    elif (flag == "W-A"): # walmart-amazon dataset
        sim_vector = wa_vector(tup1, tup2)
    elif (flag == "A-G"): # amazon-google products dataset
        sim_vector = ag_vector(tup1, tup2)
    else: # default dataset
        sim_vector = default_vector(tup1, tup2)
    
    return sim_vector