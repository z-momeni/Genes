import copy_reg
import types
import multiprocessing

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)
##Imports:
import public
import random
import Filters
import numpy as np
from math import sqrt
from sklearn import ensemble
from functools import reduce
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict

## Load Data:
X , Y = public.LoadPartOfData(c=2)
print "data is ready"

## Filter sum featur by pearson
filterdCols = Filters.myPearson(X , Y , 8000)
filterdCols.sort()
#print "selected : ",filterdCols 

if (X.shape[0]/100) > 1:
    p = int(round(X.shape[0]/100))
else:
    p = 1

splitDs = public.SpliteData(p,len(Y))
print "start proccec:"


## Def Class:
class Individual(object):
    def __init__(self , chromosome = None , feature = None):
        
        self.scores = [0,0,0]
        self.mad = 0
        if chromosome == None:
            self.feature , self.chromosome = self._makechromosome()
        else:
            self.feature , self.chromosome = feature , chromosome

    def _makechromosome(self):
        global filterdCols
        numOfFeature = random.randint(10,100)
        selected  = np.random.choice(filterdCols, numOfFeature, replace=False)
        selected.sort()
        selected = selected.tolist()
        chromosome = []
        i = 0
        for item in filterdCols:
            if i == numOfFeature:
                chromosome.append('0')
            else:
                if item == selected[i]:
                    chromosome.append('1')
                    i=i+1
                else:
                    chromosome.append('0')  
        return selected , chromosome
    
    def Fitness(self):
        global splitDs
        scores = map(self.Evalution,splitDs)
        sumScores = reduce(lambda a,b : [a[0]+b[0],a[1]+b[1],a[2]+b[2],a[3]+b[3]],scores) 
        self.mad = sumScores[0]/len(splitDs)
        self.scores[0] = sumScores[1]/len(splitDs)
        self.scores[1] = sumScores[2]/len(splitDs)
        self.scores[2] = sumScores[3]/len(splitDs)
        return self.mad,self.scores,len(self.feature)
    
    
    def Evalution(self,rows):
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
        global X , Y
        y_new = Y.iloc[rows]
        x_new = X.iloc[rows,self.feature]
        Params = {'n_estimators': 300, 'max_depth': 4, 'min_samples_split': 2,
                  'subsample':0.6, 'verbose':0, 'warm_start':True, 'alpha':0.6,
                  'learning_rate': 0.03, 'loss': 'lad'}
        estm = ensemble.GradientBoostingRegressor(**Params)

        pred = cross_val_predict(estm, x_new, y_new, cv = 3)
        result = calculator(y_new,pred,len(pred))
        return result
        
    def Crossover(self,other):
        global filterdCols
        def makeFeatureList(c,f):
            t = list()
            for i in range(len(c)):
                if c[i]=='1':
                    t.append(f[i])
            return t

        child1 , child2 = list(),list()
        for i in range(len(filterdCols)):
            if self.chromosome[i] != other.chromosome[i]:
                coin = random.choice([0,1])
                if coin == 1:
                    child1.append(self.chromosome[i])
                    child2.append(other.chromosome[i])
                else:
                    child1.append(other.chromosome[i])
                    child2.append(self.chromosome[i])
            else:
                child1.append(self.chromosome[i])
                child2.append(self.chromosome[i])
        
        f1 , f2 = makeFeatureList(child1,filterdCols),makeFeatureList(child2,filterdCols)   
        return Individual(child1,f1),Individual(child2,f2)
   
    
    def Mutation(self):
        global filterdCols
        def makeFeatureList(c,t):
            f = list()
            for i in range(len(c)):
                if c[i]=='1':
                    f.append(t[i])
            return f
        
        for i in range(2):
            coin = random.choice([0,1])
            if coin == 0: #mutate a "0"
                res_list = [i for i, value in enumerate(self.chromosome) if value == '0']
                ind = np.random.choice(res_list, 1).tolist()
                self.chromosome[ind[0]] = '1' 
                temp = filterdCols[:]
                self.feature =  makeFeatureList(self.chromosome,temp) 
                
            else: #mutate a "1"
                ind = np.random.choice(self.feature, 1).tolist()
                self.feature.remove(ind[0])
                i = 0
                numOfFeature = len(self.feature)
                chromosome = list()
                for item in filterdCols:
                    if i == numOfFeature:
                        chromosome.append('0')
                    else:
                        if item == self.feature[i]:
                            chromosome.append('1')
                            i=i+1
                        else:
                            chromosome.append('0')
                self.chromosome = chromosome
        return self
        
    

