import numpy as np
import torch
from scipy import sparse
import os.path
import time
import torch.nn as nn
from ase import Atoms, Atom

#from rdkit.Contrib.SA_Score.sascorer import calculateScore
#from rdkit.Contrib.SA_Score.sascorer
#import deepchem as dc

N_atom_features = 34 #28


def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "No running" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        #print('nvidia-smi -i '+str(i)+' | grep "No running" | wc -l > empty_gpu_check')
        if int(output)==1:
            empty.append(i)
    if len(empty)<ngpus:
        print ('avaliable gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd+=str(empty[i])+','
    return cmd

def initialize_model(model, device, load_save_file=False):
    if load_save_file:
        model.load_state_dict(torch.load(load_save_file)) 
        print('use loaded model from {}'.format(load_save_file))
    else:
        for param in model.parameters():
            if param.dim() == 1:
                continue
                nn.init.constant(param, 0)
            else:
                #nn.init.normal(param, 0.0, 0.15)
                nn.init.xavier_normal_(param)

    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    model.to(device)
    return model

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    #print list((map(lambda s: x == s, allowable_set)))
    return list(map(lambda s: x == s, allowable_set))
    # allowable set에 포함되어 있지 않으면 Exception 
    # 결과적으로 all True 값이 나와야겠네? 

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    # 허용되지 않은 입력을 마지막 요소로 매핑
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
    # allowable_set에 포함되어 있지 않으면 해당 셋의 마지막 요소로 매핑. 
    # 여기서 인자 s는 allowable set의 하나하나의 요소로서 들어감. 
    # 그 요소들이 앞에서 x로 들어온 함수의 인풋인자와 비교되어 
    # 동일한지 다른지에 대한 bool값의 list가 반환되는 듯

####
# 람다 예시 
# lambda 인자 : 표현식 >>> (lambda x,y: x + y)(10, 20)
# map 예시
# map(함수, 리스트)
#
# map(lambda x: x ** 2, range(5))             # 파이썬 2
# >>> [0, 1, 4, 9, 16]  
# list(map(lambda x: x ** 2, range(5)))     # 파이썬 2 및 파이썬 3
# >>> [0, 1, 4, 9, 16]
# 
####

##### atom feature embedding 방법 
def atom_feature(m, atom_i, i_donor, i_acceptor):

    atom = m.GetAtomWithIdx(atom_i) # m(mol)에서 atom_i 번째 atom에 대해서 atom으로 받음  
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +     # -> one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 얘는 error를 내버리네
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +  # 얘네는 이 값보다 더 크면 그냥 [-1] 로 취급해 주는데 
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) +
                    [atom.GetIsAromatic()])    # (10, 11, 5, 7, 1) --> total 34                #->(10, 6, 5, 6, 1) --> total 28
    # 그럼 mol 하나안의 atom 한개에 대해 np.array 하나를 만드는 건가. 
    # np.array(_ + _ + ...) >>> 이런 식이면 ,, -> 이런식으로 나옴 (어떤 값, ) 즉, row의 갯수가 어떤값만큼 있게 됨. 
    # (28, )

    # 근데 어째 ligand , protein label 나누는 게 없는데? 
    # 일단 계속 보자 

    
