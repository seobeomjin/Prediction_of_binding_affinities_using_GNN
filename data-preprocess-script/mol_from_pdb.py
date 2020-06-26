"""
2020 .06. 04 
Author : Noah Seo 

"""

# first method

import pickle 
import pandas as pd
import numpy as np 

from rdkit.Chem import rdmolfiles, SDMolSupplier
import os

def input_file(path):
    """Check if input file exists.""" 

    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise IOError('File {} does not exists.'.format(path))
    return path 

def output_file(path):
    """Check if output file can be created."""

    path = os.path.abspath(path)
    dirname = os.path.dirname(path)

    if not os.access(dirname, os.W_OK): #checks for Writability
        raise IOError('File {} cannot be created (check your permission)'.format(path))
    
    return path 

def string_bool(s):
    s = s.lower()
    if s in ['true','t','1','yes','y']:
        return True 
    elif s in ['false','f','0','no','n']:
        return False 
    else : 
        raise IOError('{} cannot be interpreted as bool type.'.format(s))

import argparse 
parser = argparse.ArgumentParser()

parser.add_argument('--ligand', '-l', required=True, type=input_file, nargs='+', help='please input ligand structure files')
parser.add_argument('--pocket', '-p', required=True, type=input_file, nargs='+', help='please input pocket structure files')
parser.add_argument('--affinities', '-a', default=None, type=input_file, help='please input affinity csv file to predict binding affinities')
parser.add_argument("--test", default=None , type=str, nargs="*", help="enter the protein targets which want to set as test target")
parser.add_argument('--output','-o', default='./complex.pkl', type=output_file)
parser.add_argument('--test_output', default='./test_keys.pkl', type=output_file)
parser.add_argument('--nonetype', default=None, type=str)
parser.add_argument('--verbose', '-v', default=True, help='print helpful description')
args = parser.parse_args()

num_pockets = len(args.pocket)
num_ligands = len(args.ligand)
if num_pockets !=1 and num_pockets != num_ligands:
    raise IOError('%s pockets specified for %s ligands. You must either provide'
                  'a single pocket or a separate pocket for each ligand'
                  % (num_pockets, num_ligands))
if args.verbose: 
    print('%s ligands and %s pockets prepared :' %(num_ligands, num_pockets))
    if num_pockets == 1:
        print('pocket: %s' % args.pocket[0])
        for ligand_file in args.ligand:
            print('ligand: %s' %ligand_file)
    else: 
        for ligand_file, pocket_file in zip(args.ligand, args.pocket):
            print(' ligand: %s, pocket: %s' %(ligand_file, pocket_file))
    print()


def make_pairs(pocket, ligand, aff_csv, test_target, output, test_output):
    
    num_pockets = len(pocket)
    num_ligands = len(ligand)

    aff_df = pd.read_csv(aff_csv)

    NoneType_pdbid = []

    if test_target !=None : 
        
        test_pdbid = aff_df[np.in1d(aff_df['protein_name'],test_target)]['pdbid']
        test_pdbid_list = list(test_pdbid)

        train_pdbid = aff_df[~np.in1d(aff_df['protein_name'],test_target)]['pdbid']
        train_pdbid_list = list(train_pdbid)
        
        test_output = test_output
        #train_output = "train_keys"
        test_pairs = ()
        test_pairs_list = []
        train_pairs = ()
        train_pairs_list = []

        if num_pockets == 1 : 
            # pocket data
            pocket_mol = rdmolfiles.MolFromPDBFile(pocket[0])
            for ligand_file in ligand: 
                # affinity data 
                ligand_filename = os.path.splitext(os.path.split(ligand_file)[1])[0]
                pdbid = ligand_filename.split('_')[0]
                aff = float(aff_df.loc[np.in1d(aff_df['pdbid'], pdbid),'-logKd/Ki'])

                # ligand data 
                pocket_datatype = os.path.splitext(os.path.split(pocket[0])[1])[1]
                ligand_datatype = os.path.splitext(os.path.split(ligand_file)[1])[1]

                if ligand_datatype == '.mol2': 
                    ligand_mol = rdmolfiles.MolFromMol2File(ligand_file)
                    if ligand_mol is not None and pocket_mol is not None : 
                        if pdbid in train_pdbid_list :
                            train_pairs = ligand_mol, pocket_mol, aff 
                            train_pairs_list.append(train_pairs)
                        elif pdbid in test_pdbid_list : 
                            test_pairs = ligand_mol, pocket_mol, aff 
                            test_pairs_list.append(test_pairs)
                        else : 
                            raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file")
                    else : 
                        NoneType_pdbid.append(pdbid)
                        print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                        continue
                
                elif ligand_datatype == '.sdf':
                    ligand_mol = SDMolSupplier(ligand_file)
                    for lgd_mol in ligand_mol:
                        if lgd_mol is not None and pocket_mol is not None :
                            if pdbid in train_pdbid_list :
                                train_pairs = lgd_mol, pocket_mol, aff 
                                train_pairs_list.append(train_pairs)
                            elif pdbid in test_pdbid_list : 
                                test_pairs = lgd_mol, pocket_mol, aff 
                                test_pairs_list.append(test_pairs)
                            else : 
                                raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file") 
                        else :
                            NoneType_pdbid.append(pdbid)
                            print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                            continue
                            #raise IOError("ligand_mol is None from SDMolSupplier(ligand.sdf). check the file.")
        else : 
            for pocket_file, ligand_file in zip(pocket, ligand):
            
                # affinity data 
                ligand_filename = os.path.splitext(os.path.split(ligand_file)[1])[0]
                pdbid = ligand_filename.split('_')[0]
                aff = float(aff_df.loc[np.in1d(aff_df['pdbid'], pdbid),'-logKd/Ki'])

                #pocket data
                pocket_mol = rdmolfiles.MolFromPDBFile(pocket_file)

                #ligand data 
                pocket_datatype = os.path.splitext(os.path.split(pocket[0])[1])[1]
                ligand_datatype = os.path.splitext(os.path.split(ligand_file)[1])[1]
                
                if ligand_datatype == '.mol2': 
                    ligand_mol = rdmolfiles.MolFromMol2File(ligand_file)
                    if ligand_mol is not None and pocket_mol is not None : 
                        if pdbid in train_pdbid_list :
                            train_pairs = ligand_mol, pocket_mol, aff 
                            train_pairs_list.append(train_pairs)
                        elif pdbid in test_pdbid_list : 
                            test_pairs = ligand_mol, pocket_mol, aff 
                            test_pairs_list.append(test_pairs)
                        else : 
                            raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file")
                    else : 
                        NoneType_pdbid.append(pdbid)
                        print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                        continue
                elif ligand_datatype == '.sdf':
                    ligand_mol = SDMolSupplier(ligand_file)
                    for lgd_mol in ligand_mol:
                        if lgd_mol is not None and pocket_mol is not None :
                            if pdbid in train_pdbid_list :
                                train_pairs = lgd_mol, pocket_mol, aff 
                                train_pairs_list.append(train_pairs)
                            elif pdbid in test_pdbid_list : 
                                test_pairs = lgd_mol, pocket_mol, aff 
                                test_pairs_list.append(test_pairs)
                            else : 
                                raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file") 
                        else :
                            NoneType_pdbid.append(pdbid)
                            print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                            continue
                            #raise IOError("ligand_mol is None from SDMolSupplier(ligand.sdf). check the file.")                          
        with open(test_output,'wb') as f:
            pickle.dump(test_pairs_list, f) 
        with open(output, 'wb') as f : 
            pickle.dump(train_pairs_list, f)
        with open("Nonetype.txt",'w') as f : 
            f.write(str(NoneType_pdbid))
        print("\n created number of train set : {} , created number of test set : {}".format(len(train_pairs_list), len(test_pairs_list)))
        print("NoneType data list is saved in the text file. total number of missed data is {}".format(len(NoneType_pdbid)))
        
    else : 
        test_pdbid = aff_df[np.in1d(aff_df['dataset'],'test')]['pdbid']
        test_pdbid_list = list(test_pdbid)

        train_pdbid = aff_df[np.in1d(aff_df['dataset'],'train')]['pdbid']
        train_pdbid_list = list(train_pdbid)
        
        test_output = test_output
        #train_output = "train_keys"
        test_pairs = ()
        test_pairs_list = []
        train_pairs = ()
        train_pairs_list = []

        if num_pockets == 1 : 
            # pocket data
            pocket_mol = rdmolfiles.MolFromPDBFile(pocket[0])
            for ligand_file in ligand: 
                # affinity data 
                ligand_filename = os.path.splitext(os.path.split(ligand_file)[1])[0]
                pdbid = ligand_filename.split('_')[0]
                aff = float(aff_df.loc[np.in1d(aff_df['pdbid'], pdbid),'-logKd/Ki'])

                # ligand data 
                pocket_datatype = os.path.splitext(os.path.split(pocket[0])[1])[1]
                ligand_datatype = os.path.splitext(os.path.split(ligand_file)[1])[1]

                if ligand_datatype == '.mol2': 
                    ligand_mol = rdmolfiles.MolFromMol2File(ligand_file)
                    if ligand_mol is not None and pocket_mol is not None : 
                        if pdbid in train_pdbid_list :
                            train_pairs = ligand_mol, pocket_mol, aff 
                            train_pairs_list.append(train_pairs)
                        elif pdbid in test_pdbid_list : 
                            test_pairs = ligand_mol, pocket_mol, aff 
                            test_pairs_list.append(test_pairs)
                        else : 
                            raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file")
                    else : 
                        NoneType_pdbid.append(pdbid)
                        print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                        continue
                
                elif ligand_datatype == '.sdf':
                    ligand_mol = SDMolSupplier(ligand_file)
                    for lgd_mol in ligand_mol:
                        if lgd_mol is not None and pocket_mol is not None :
                            if pdbid in train_pdbid_list :
                                train_pairs = lgd_mol, pocket_mol, aff 
                                train_pairs_list.append(train_pairs)
                            elif pdbid in test_pdbid_list : 
                                test_pairs = lgd_mol, pocket_mol, aff 
                                test_pairs_list.append(test_pairs)
                            else : 
                                raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file") 
                        else :
                            NoneType_pdbid.append(pdbid)
                            print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                            continue
                            #raise IOError("ligand_mol is None from SDMolSupplier(ligand.sdf). check the file.")
        else : 
            for pocket_file, ligand_file in zip(pocket, ligand):
            
                # affinity data 
                ligand_filename = os.path.splitext(os.path.split(ligand_file)[1])[0]
                pdbid = ligand_filename.split('_')[0]
                aff = float(aff_df.loc[np.in1d(aff_df['pdbid'], pdbid),'-logKd/Ki'])

                #pocket data
                pocket_mol = rdmolfiles.MolFromPDBFile(pocket_file)

                #ligand data 
                pocket_datatype = os.path.splitext(os.path.split(pocket[0])[1])[1]
                ligand_datatype = os.path.splitext(os.path.split(ligand_file)[1])[1]
                
                if ligand_datatype == '.mol2': 
                    ligand_mol = rdmolfiles.MolFromMol2File(ligand_file)
                    if ligand_mol is not None and pocket_mol is not None : 
                        if pdbid in train_pdbid_list :
                            train_pairs = ligand_mol, pocket_mol, aff 
                            train_pairs_list.append(train_pairs)
                        elif pdbid in test_pdbid_list : 
                            test_pairs = ligand_mol, pocket_mol, aff 
                            test_pairs_list.append(test_pairs)
                        else : 
                            raise IOError("this pdbid, %s , is not included in any part, train or test. check the each pdb list or file" %pdbid)
                    else : 
                        NoneType_pdbid.append(pdbid)
                        print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                        continue
                elif ligand_datatype == '.sdf':
                    ligand_mol = SDMolSupplier(ligand_file)
                    for lgd_mol in ligand_mol:
                        if lgd_mol is not None and pocket_mol is not None :
                            if pdbid in train_pdbid_list :
                                train_pairs = lgd_mol, pocket_mol, aff 
                                train_pairs_list.append(train_pairs)
                            elif pdbid in test_pdbid_list : 
                                test_pairs = lgd_mol, pocket_mol, aff 
                                test_pairs_list.append(test_pairs)
                            else : 
                                raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file") 
                        else :
                            NoneType_pdbid.append(pdbid)
                            print("%s ligand is NoneType. ignore it from dataset." % pdbid)
                            continue
                            #raise IOError("ligand_mol is None from SDMolSupplier(ligand.sdf). check the file.")                          
        with open(test_output,'wb') as f:
            pickle.dump(test_pairs_list, f) 
        with open(output, 'wb') as f : 
            pickle.dump(train_pairs_list, f)
        with open("Nonetype_.txt",'w') as f : 
            f.write(str(NoneType_pdbid))
        print("\n created number of train set : {} , created number of test set : {}".format(len(train_pairs_list), len(test_pairs_list)))
        print("NoneType data list is saved in the text file. total number of missed data is {}".format(len(NoneType_pdbid)))
        
        """
        pair_tuple = ()
        pair_list = []
        if num_pockets == 1 : 
            pocket_mol = rdmolfiles.MolFromPDBFile(pocket[0])
            for ligand_file in ligand: 
                # affinity data 
                ligand_filename = os.path.splitext(os.path.split(ligand_file)[1])[0]
                pdbid = ligand_filename.split('_')[0]
                aff = float(aff_df.loc[np.in1d(aff_df['pdbid'], pdbid),'-logKd/Ki'])
                
                #ligand data 
                pocket_datatype = os.path.splitext(os.path.split(pocket[0])[1])[1]
                ligand_datatype = os.path.splitext(os.path.split(ligand_file)[1])[1]
                
                if ligand_datatype == '.mol2': 
                    ligand_mol = rdmolfiles.MolFromMol2File(ligand_file)
                    if pdbid in train_pdbid_list :
                        train_pairs = ligand_mol, pocket_mol, aff 
                        train_pairs_list.append(train_pairs)
                    elif pdbid in test_pdbid_list : 
                        test_pairs = ligand_mol, pocket_mol, aff 
                        test_pairs_list.append(test_pairs)
                    else : 
                        raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file")
                elif ligand_datatype == '.sdf':
                    ligand_mol = SDMolSupplier(ligand_file)
                    for lgd_mol in ligand_mol:
                        if lgd_mol is not None:
                            if pdbid in train_pdbid_list :
                                train_pairs = lgd_mol, pocket_mol, aff 
                                train_pairs_list.append(train_pairs)
                            elif pdbid in test_pdbid_list : 
                                test_pairs = lgd_mol, pocket_mol, aff 
                                test_pairs_list.append(test_pairs)
                            else : 
                                raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file") 
                        else : 
                            raise IOError("ligand_mol is None from SDMolSupplier(ligand.sdf). check the file.")
        else : 
            for pocket_file, ligand_file in zip(pocket, ligand):
                
                # affinity data 
                ligand_filename = os.path.splitext(os.path.split(ligand_file)[1])[0]
                pdbid = ligand_filename.split('_')[0]
                aff = float(aff_df.loc[np.in1d(aff_df['pdbid'], pdbid),'-logKd/Ki'])

                # pocket data 
                pocket_mol = rdmolfiles.MolFromPDBFile(pocket_file)

                # ligand data 
                pocket_datatype = os.path.splitext(os.path.split(pocket[0])[1])[1]
                ligand_datatype = os.path.splitext(os.path.split(ligand_file)[1])[1]
                
                if ligand_datatype == '.mol2': 
                    ligand_mol = rdmolfiles.MolFromMol2File(ligand_file)
                    if pdbid in train_pdbid_list :
                        train_pairs = ligand_mol, pocket_mol, aff 
                        train_pairs_list.append(train_pairs)
                    elif pdbid in test_pdbid_list : 
                        test_pairs = ligand_mol, pocket_mol, aff 
                        test_pairs_list.append(test_pairs)
                    else : 
                        raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file")
                elif ligand_datatype == '.sdf':
                    ligand_mol = SDMolSupplier(ligand_file)
                    for lgd_mol in ligand_mol:
                        if lgd_mol is not None:
                            if pdbid in train_pdbid_list :
                                train_pairs = lgd_mol, pocket_mol, aff 
                                train_pairs_list.append(train_pairs)
                            elif pdbid in test_pdbid_list : 
                                test_pairs = lgd_mol, pocket_mol, aff 
                                test_pairs_list.append(test_pairs)
                            else : 
                                raise IOError("this pdbid is not included in any part, train or test. check the each pdb list or file") 
                        else : 
                            raise IOError("ligand_mol is None from SDMolSupplier(ligand.sdf). check the file.")
        with open(output,'wb') as f:
            pickle.dump(pair_list, f)
        print("\n created number of train set : {} , created number of test set : {}".format(len(train_pairs_list), len(test_pairs_list)))
        """
    
            
if __name__ == '__main__':
    make_pairs(args.pocket, args.ligand, args.affinities, args.test, args.output, args.test_output)
    #if args.verbose:
    #    print('\n\ncreated %s with %s structures' % (args.output, num_ligands))

    # 하나의 pickle file 안에 m1, m2 가 다 들어있어야 함  
    # 튜플은 리스트와 동일하지만, 값의 변경이 불가능함.
