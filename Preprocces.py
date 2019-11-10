import csv
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler


def getPreProcData():
    
    filename = ['DS097_270','DS236_270','DS242_270','DS317_270',
                'DS642_270','DS869_270','DS257_270','DS008_270', 
                'DS149_450','DS169_450','DS638_450','DS812_270',
                'DS064_450','DS870_450','DS128_450','DS279_450'] 

    XTrain_Frames = []
    YTrain_Frames = []

    for DS in filename:
    #load data:
        pathname = "Datasets/"+ DS +".csv"
        X , Y = getDataFromOrginalDS(pathname,xx_range=1, yy_range=1, 
                                                                    xy_range=0, startCol=1, header=0, sep=',') 

        print "DataSet ",DS," : ","[ column:",X.shape[0]," , Row:",X.shape[1],"]"
        X = X.transpose()
        
        XTrain_Frames.append(pd.DataFrame(X)) 
        YTrain_Frames.append(Y)
    
    xtrain_all = pd.concat(XTrain_Frames)
    ytrain_all = pd.concat(YTrain_Frames)
    
    X = Imput(xtrain_all, ytrain_all)
    X = Robust_StandardAll(X)
    X = MinMaxAll(X)
    X = pd.DataFrame(X)
    print "[ column:",X.shape[1]," , Row:",X.shape[0],"]"
    ind = [i for i in range(X.shape[1])]
    #splite-------------------------------------------------
    esm = []
    X["lable"] = ytrain_all.tolist()
    result = X.sort_values(by=['lable'])
    result.to_csv('MergedDs.csv', index = None, header=True)
    print "done"
    
getPreProcData()
      
  
################################################################################################  

def getDataFromOrginalDS(pathName, xx_range, yy_range, xy_range, startCol, header=-1, sep=','):
    mydata = pd.read_csv(pathName,header,sep)
    #for take columns that we want to be in x
    yx_range = []    
    for i in range(startCol,mydata.shape[1]):
        yx_range.append(i)

    #iloc[x_range,y_range]
    """this setting is for data that has class lable in first column, and in first row has index number that 
    # with header = -1 removed."""
    x_data = mydata.iloc[xx_range: , yx_range]
    y_data = mydata.iloc[xy_range , yy_range:]
    return (x_data,y_data)

def MinMaxAll(x_train):
    min_max_scaler = MinMaxScaler()
    x_train_minmax = min_max_scaler.fit_transform(x_train)
    return x_train_minmax


def Robust_StandardAll(x_train):
    scaler = RobustScaler().fit(x_train)
    xNormal_train = scaler.transform(x_train)
    return (xNormal_train)

def Imput(x,y,strgy = 'mean'):
    imp = SimpleImputer(missing_values=np.nan, strategy=strgy)
    x_impute = imp.fit_transform(x,y)
    return x_impute
