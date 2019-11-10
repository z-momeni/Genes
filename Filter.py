import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def myPearson(xFrame,y,NumOfSlc):
    
    pearsonVal = []
    col = [i for i in range(xFrame.shape[1])]
    for i in col:
        temp = xFrame.iloc[:,i]
        corr , p_val = pearsonr(temp, y)
        pearsonVal.append(abs(corr))

    np_pearsonVal = np.array(pearsonVal)
    ind = np.argsort(np_pearsonVal)[-NumOfSlc:][::-1]   
    
    return ind

