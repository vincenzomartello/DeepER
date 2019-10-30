# Utility per fare il dump di una lista su disco
import pickle

def save_list(lista, name):
    with open(name, 'wb') as f:
        pickle.dump(lista, f)


def load_list(list_file):
    with open(list_file, 'rb') as f:
        return pickle.load(f)
