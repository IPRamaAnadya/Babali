import numpy as np
import librosa.feature

class Extraction(object):
    def __init__(self,sr = 22050):
        self.sr = sr
    def get(self,x):
        mfcc = librosa.feature.mfcc(x, n_mfcc = 13)
        mfcc = np.delete(mfcc,0,0)
        return np.ndarray.flatten(mfcc)