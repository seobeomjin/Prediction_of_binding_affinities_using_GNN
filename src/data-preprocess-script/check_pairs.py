
import argparse
import os 

def input_file(path):
    """Check if input file exists.""" 

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File {} does not exists.'.format(path))
    return path 

parser = argparse.ArgumentParser()
parser.add_argument("--dir", required=True, type=input_file, nargs='+')
args = parser.parse_args()

path =  args.dir

for f in path : 
    pdbid = os.path.split(f)[1]

    ligand_path = f + '/'+ pdbid + '_ligand.mol2'
    pocket_path = f + '/'+ pdbid + '_pocket.pdb'

    if os.path.isfile(ligand_path) and os.path.isfile(pocket_path): 
        if f == path[-1]:
            print("not found")
        continue
    else : 
        #print("some file is missed in this pdbid")
        print(pdbid)

# usage -> check the structural file length
# >>> python check_pairs.py --dir data/whole_set/refined-set/*
# ik6p from refined set is deleted because it had no ligand mol2 file format