from python_speech_features import mfcc
import scipy
from scipy import io
from scipy.io import wavfile
import glob
import numpy as np
import os
from chainer.datasets import tuple_dataset


def write_ceps(ceps,fn):
    base_folder_name, file_name = os.path.split(fn)
    #print(base_folder_name.replace("raw", "cep"))
    if not os.path.exists(base_folder_name.replace("raw", "cep")):
        os.makedirs(base_folder_name.replace("raw", "cep"))
    base_fn,ext = os.path.splitext(fn)
    data_fn = base_fn.replace("raw", "cep") + ".ceps"
    np.save(data_fn,ceps)

def create_ceps(fn):
    (rate, X) = io.wavfile.read(fn)
    ceps = mfcc(X, rate, 0.0001, 0.0001, 52, 104, 4096)
    #print(ceps)
    isNan = False
    for num in ceps:
        if np.isnan(num[1]):
            isNan = True
    if isNan == False:
        write_ceps(ceps,fn)

def read_ceps(name_list, is_test = True, base_dir = os.getcwd()):
    X, y = [],[]
    test_filename_str1 = "Test"
    test_filename_str2 = "_test"
    if not is_test:
        test_filename_str1 = "Train"
        test_filename_str2 = ""
    for label,name in enumerate(name_list):
        for fn in glob.glob(os.path.join(base_dir, "cepData", "cep" + test_filename_str1 + "Data", name + test_filename_str2, "*.ceps.npy")):
            ceps = np.load(fn)
            num_ceps = len(ceps)
            X.append(np.mean(ceps[:],axis=0))
            y.append(label)
            #print(np.mean(ceps[:],axis=0)[11], label)
    #print(X)
    return np.array(X),np.array(y)
