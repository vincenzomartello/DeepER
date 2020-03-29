from .data import data2Inputs_unlabeled


def dataset_tolist(dataset,lprefix,rprefix,ignore_columns=['label','id']):
    res = []
    attributes = [col for col in dataset.columns if col not in ignore_columns]
    for i in range(len(dataset.index)):
        row = dataset.iloc[i]
        ltuple,rtuple = ([],[])
        for attr in attributes:
            if attr.startswith(lprefix):
                ltuple = ltuple+ str(row[attr]).split()
            else:
                rtuple = rtuple+str(row[attr]).split()
        res.append((ltuple,rtuple))
    return res


def wrap_deeper(dataset,lprefix,rprefix,model,tokenizer,embedding_model,ignore_columns=['label','id']):
    dataset_list = dataset_tolist(dataset,lprefix,rprefix,ignore_columns=ignore_columns)
    ltokens,rtokens = data2Inputs_unlabeled(dataset_list,tokenizer)
    lembeddings,rembeddings = embedding_model.predict([ltokens,rtokens])
    predictions = model.predict([lembeddings,rembeddings])
    return predictions