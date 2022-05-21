import numpy as np
import noisereduce as nr

class Preprocessing(object):
    def __init__(self,sr = 22050):
        self.sr=sr
    def get(self,x):
        x = self.normalize_sample(x)
        x =self.noise_reduce(x)
        return x
        
    def normalize_sample(self,x):
        x -= np.mean(x)
        return x

    def noise_reduce(self, x):
        x = nr.reduce_noise(y=x, sr=self.sr)
        return x