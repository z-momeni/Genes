##Imports:
import numpy as np
import pandas as pd
from math import sqrt
from sklearn import ensemble
from sklearn.metrics import r2_score

## Load Data:
#[ column: 21291  , Row: 2079 ]

def LoadPartOfData(c):
    data = pd.read_csv("Train3Class.csv",header = 0,sep = ',')
    data = data.sample(frac=1)
    classMask = data["class"]==c
    filterd = data[classMask]
    y = filterd.loc[:, "lable"]
    x = filterd.drop(["lable","class"], axis=1)
    return x,y
  
#X,Y = LoadPartOfData(0)
#print X
#print Y

def LoadPartOfDataTest(f,c):
    data = pd.read_csv("Test3Class.csv",header = 0,sep = ',')
    data = data.sample(frac=1)
    classMask = data["prediction"]==c
    filterd = data[classMask]
    y = filterd.loc[:, "lable"]
    x = filterd.drop(["lable","class","prediction"], axis=1)
    x = x.iloc[:,f]
    return x,y

def getXData(cols):
    data = pd.read_csv("outX.csv",header = 0,sep = ',')   
    x = data.iloc[:, cols]
    return x

#a = getXData()
#print a

def getYData():
    data = pd.read_csv("outY.csv",header = -1,sep = ',')   
    y = data.iloc[:,1:] 
    return y

#b = getYData()
#print b

def LoadData():
    data = pd.read_csv("out.csv",header = 0,sep = ',') 
    y = data.loc[:, "lable"]
    x = data.drop(["lable"], axis=1)
    return x,y

#x , y = LoadData()
# print x
# print x.shape[1] #Col 21291
# print x.shape[0] #Row 2079
# print y #2079

def generateSubFeature(n,k):
    subs = list()
    for i in range(n):
        sub = np.random.choice(range(21291), k, replace=False)
        sub.sort()
        subs.append(sub.tolist())
    return subs

#a = generateSubFeature(30,2000)


def SpliteData(num_part,records):
    arr = list()
    for p in range(num_part):
        arr.append(list())
    k = 0
    for i in range(records):
        if k < num_part:
            arr[k].append(i)
            k = k + 1
        else:
            k = 0
            arr[k].append(i)
            k = k + 1
    return arr
  
def calculator(y,y_pred,m):
    temp1 = 0
    temp2 = 0
    y = y.tolist()

    for i in range(m):
        dis = y[i] - y_pred[i]
        temp1 = temp1 + abs(dis)
        temp2 = temp2 + dis**2
    
    MAD = temp1/m
    MSE = temp2/m
    RMSE = sqrt(MSE)
    Rsquared = r2_score(y, y_pred)
    ans = [MAD,MSE,RMSE,Rsquared]
    return ans
  
def EvaluationTest(xt,yt,f,c):
    Params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 2,
            'subsample':0.6, 'verbose':0, 'warm_start':True, 'alpha':0.6,
            'learning_rate': 0.03, 'loss': 'lad'}
    clf = ensemble.GradientBoostingRegressor(**Params)
    x_train , y_train = LoadPartOfData(c=c)
    x_train = x_train.iloc[:,f]
    clf.fit(x_train , y_train)
    pred = clf.predict(xt)
    ansTest = calculator(yt,pred,len(pred))
    return ansTest
