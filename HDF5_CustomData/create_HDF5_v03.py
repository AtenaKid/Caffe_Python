from __future__ import print_function
import pandas as pd
import  os

import scipy.io as sio
import numpy as np 
import h5py
import math

def load_data(file_name):
    '''
    Function to read data from file_name.mat file 
    the mat file should include data L_SQ, L_RDOQ_LE, 
    in compressed mat file (i.e not in -v.73 version)
    '''
    content = sio.loadmat(file_name + '.mat')
    L_SQ = np.asarray(content['L_SQ'])
    L_SQDZ = np.asarray(content['L_SQDZ'])
    L_RDOQ_LE = np.asarray(content['L_RDOQ_LE'])
    #L_RDOQ_LE = np.asarray(L_RDOQ[0, 0])
    Res_RDOQ = abs(L_SQ) - abs(L_RDOQ_LE)
    
    data =abs(L_SQ)
    res = Res_RDOQ

    return data, res 

def create_hdf5(mode, train_path, file_name ):
    data, res = load_data(file_name)
    tmp = data.shape
    tu_size = int(math.sqrt(tmp[1]))
    no_tu = tmp[0]

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    image = np.zeros([no_tu, tu_size, tu_size, 1])
    label = image 
    for idx in range(0, no_tu):
        if (idx % 10000) == 0:
            print("Process: ", idx * 1.0 /no_tu * 100.0)
        image[idx, :, :, 0] = data[idx, :].reshape(tu_size, tu_size)
        label[idx, :, :, 0] = res[idx, :].reshape(tu_size, tu_size)

    root_path = train_path
    with h5py.File(root_path + mode + '.h5', 'w') as f:
        f['data'] = image
        f['label'] = label

    with open('./list.txt', 'a') as f:
        f.write(root_path + mode + '.h5\n')

if __name__ == '__main__':
    save_path = '/Research_RDOQ/1DataPreparation/GenTrainData_Python/'
    mode = 'train'
    create_hdf5(mode, save_path, 'test')
    