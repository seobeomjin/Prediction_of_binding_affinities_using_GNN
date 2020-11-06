import argparse 
import os 
from rdkit import Chem 
import pickle

def input_file(path):
    #Check if input file exists.

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File {} does not exists.'.format(path))
    return path 

parser = argparse.ArgumentParser()
parser.add_argument("--train_keys",type=input_file, default='data/train_keys.pkl')
parser.add_argument("--test_keys",type=input_file, default='data/test_keys.pkl')
args = parser.parse_args()


    ################ check the longest item[H] ################  

with open (args.train_keys, 'rb') as fp:
    train_keys = pickle.load(fp)
with open (args.test_keys, 'rb') as fp:
    test_keys = pickle.load(fp)

max_natoms = 0

for i, key in enumerate(train_keys):
    m1, m2, _ = key
    if m1 is not None and m2 is not None : 
        atom_sum = m1.GetNumAtoms() + m2.GetNumAtoms()
        if atom_sum >= max_natoms : 
            max_natoms = atom_sum 
        else : 
            continue  
for i, key in enumerate(test_keys):
    m1, m2, _ = key
    if m1 is not None and m2 is not None : 
        atom_sum = m1.GetNumAtoms() + m2.GetNumAtoms()
        if atom_sum >= max_natoms : 
            max_natoms = atom_sum 
        else : 
            continue

print("max_natoms : {}".format(max_natoms))