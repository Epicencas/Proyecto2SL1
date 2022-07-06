from select import select
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import re
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



### ============ Imputación de Variabel Numéricas ========
class NumericalImputerOperator(BaseEstimator, TransformerMixin):
    
    def __init__(self, imputerType = 'mean', varNames = None):
        self.varNames = varNames
        if(imputerType == 'mean'):
            self.imputerType = 'mean'
        elif(imputerType == 'median'):
            self.imputerType = 'median'
        else:
            print("Mecanismo de imptación invalido.\n")

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()
        for col in self.varNames:
            if(self.imputerType == 'mean'):
                imputerValue = np.round(X[col].mean(), 0)
            else:
                imputerValue = np.round(X[col].median(), 0)
            X[col].fillna(imputerValue, inplace=True)
        return X

    
### ============ Codificación de Variables Categoricas ========
class CategoricalEncoderOperator(BaseEstimator, TransformerMixin):

    def __init__(self, varNames = None):
        self.encoder_dict = {}
        self.varNames = varNames

    def fit(self, X, y = None) -> None:
        """
            fit para calculo de diccionario de codificación
        """
        for col in self.varNames:
            self.encoder_dict[col] = (X[col].value_counts().sort_values(ascending=False)).to_dict()
        return self
    
    def transform(self, X, y = None):
        """
            transforamción para variables, según diccionario de codificación
        """
        X = X.copy()
        for col in self.varNames:
            X[col] = X[col].map(self.encoder_dict[col])
        return X

###  ===========Frecuency Encoding con código normal ========

def executeFreqEncoding(df, map_type='freq'):
    colnames = df.columns
    for col in colnames:
        if(df[col].dtype == 'object'):
            factor_div = 1 if (map_type == 'freq') else len(df[col])
            mapper = (df[col].value_counts().sort_values(ascending=False)/factor_div).to_dict()
            df[col] = df[col].map(mapper)    

            
###  ===========Imputación con Missings en Variables Categoricas ========
def ImpVC (df):
    df[col].fillna('Missing', inplace=True)
    


    ### ============ Transformación de Variables ========
class transfvnum(BaseEstimator, TransformerMixin):

    def __init__(self, varNames = None):
        self.varNames = varNames
        

    def fit(self, X, y = None):
        for col in self.varNames:
            dataset_log = X.loc[:,[col,target]]
            dataset_log[col+"_log"] = np.log(X[col])
            corrlog = np.corrcoef(dataset_log[col+"_log"], dataset_log[target])[0,1]
    
            dataset_inv = X.loc[:,[col,target]]
            dataset_inv[col+"_inv"] = (1 / X[col])
            corrinv = np.corrcoef(dataset_inv[col+"_inv"], dataset_inv[target])[0,1]
    
            dataset_poly2 = X.loc[:,[col,target]]
            dataset_poly2[col+"_poly2"] = (X[col]**2)
            corrpoly2 = np.corrcoef(dataset_poly2[col+"_poly2"], dataset_poly2[target])[0,1]
    
            list = (corrlog, corrinv, corrpoly2)#, corrbxcx, corryeoj)
            mayor = max(list)
            pos = list.index(mayor)   

#La función indica el cual fue el valor imputado y hace la transformación
            if list.index(mayor) == 0:
              X = df.replace(df[col], np.log(df[col])) 
              #print("Transformación con Logaritmo")
            elif list.index(mayor) == 1:
              X = df.replace(df[col], 1/df[col]) 
              #print("Transformación Inversa")
            elif list.index(mayor) == 2:
              X = df.replace(df[col], df[col]**2)
              #print("Transformación Polinomial 2")
           
        return self

    def transform(self, X, y = None):
        X = X.copy()
        for col in self.varNames:
            X[col] = np.where(X[col] >= self.upper, self.upper,
                np.where(
                    X[col] < self.lower, self.lower, X[col]
                )    
            )
        return X

###  ===========Transformación de Variables========

def transfvnum (df, col, target):
#Para cada transformación se crea un dataset temporal con la columna de interés y con la que se busca una correlación. 
#Luego al dataset temporal se le añade una nueva columna con la transformación aplicada
#Por último se obtiene la correlación de la columna transformanda con la otra.
    dataset_log = df.loc[:,[col,target]]
    dataset_log[col+"_log"] = np.log(df[col]+1.1)
    corrlog = np.corrcoef(dataset_log[col+"_log"], dataset_log[target])[0,1]
    
    dataset_inv = df.loc[:,[col,target]]
    dataset_inv[col+"_inv"] = (1 / (df[col]+1))
    corrinv = np.corrcoef(dataset_inv[col+"_inv"], dataset_inv[target])[0,1]
    
    dataset_poly2 = df.loc[:,[col,target]]
    dataset_poly2[col+"_poly2"] = (df[col]**2)
    corrpoly2 = np.corrcoef(dataset_poly2[col+"_poly2"], dataset_poly2[target])[0,1]
    
    #dataset_bxcx = df.loc[:,[col,target]]
    #dataset_bxcx[col+"_bxcx"], lambdaX = stats.boxcox(df[col])
    #corrbxcx = np.corrcoef(dataset_bxcx[col+"_bxcx"], dataset_bxcx[target])[0,1]
    
    #dataset_yeoj = df.loc[:,[col,target]]
    #dataset_yeoj[col+"_yeoj"], lambdaX = stats.yeojohnson(df[col])
    #corryeoj = np.corrcoef(dataset_yeoj[col+"_yeoj"], dataset_yeoj[target])[0,1]
#Todos las correlaciones se ingresan a una lista    
    list = (corrlog, corrinv, corrpoly2)#, corrbxcx, corryeoj)
#En la variable mayor se asigna el valor máximo del listado
    mayor = max(list)
    pos = list.index(mayor)   

#La función indica el cual fue el valor imputado y hace la transformación
    if list.index(mayor) == 0:
      dataset = df.replace(df[col], np.log(df[col])) 
      #print("Transformación con Logaritmo")
    elif list.index(mayor) == 1:
      dataset = df.replace(df[col], 1/df[col]) 
      #print("Transformación Inversa")
    elif list.index(mayor) == 2:
      dataset = df.replace(df[col], df[col]**2)
      #print("Transformación Polinomial 2")
    #elif list.index(mayor) == 3:
      #dataset,lambdaX = df.replace(df[col], stats.boxcox(df[col]))
      #print("Transformación Box Cox")
    #elif list.index(mayor) == 4:
     # dataset = df.replace(df[col], stats.yeojohnson(df[col]))
      #print("Transformación Yeo Jonhson")    
   
    
###  ===========Imputación de Variables Numéricas ========

def imputVN(df, col, target):
 
#Primero se hace la validación de si apliaca o no para hacer imputación, sino ya no realiza nada más
 if df[col].isnull().mean() > 0.2:
    print("Columna no es válida para imputación") 
 else:
#Se obtienen los valores de promedio y media de la columna indicada. También se obtiene el valor máximo 
     Vpromedio = np.round(df[col].mean(), 0)
     Vmediana = np.round(df[col].median(), 0)
     Vmax = max(df[col])
     Varb= -1

#Se insertan los valores en los espacios nulos
     dfmeanImp = df[col].fillna(Vpromedio)
     dfmedianImp = df[col].fillna(Vmediana) 
     dfmaxImp = df[col].fillna(Vmax)
     dfvarbImp= df[col].fillna(Varb)
        
#Se obtiene el coeficiente de correlacion de la columna nueva contra la columna objetivo
     p = np.corrcoef(dfmeanImp, df[target])[0,1]
     m = np.corrcoef(dfmedianImp, df[target])[0,1]
     vmax = np.corrcoef(dfmaxImp, df[target])[0,1]
     varb = np.corrcoef(dfvarbImp, df[target])[0,1]

#Todos las correlaciones se ingresan a una lista    
     list = (p, m, vmax, varb)
#En la variable mayor se asigna el valor máximo del listado, con la posición (pos) se obtendra el tipo del valor imputado
     mayor = max(list)
     pos = list.index(mayor)   

#La función indica el cual fue el valor imputado y hace la imputación
     if list.index(mayor) == 0:
      dataset = df[col].fillna(Vpromedio, inplace=True)
      #print("Promedio Imputado")
      #print(Vpromedio)
     elif list.index(mayor) == 1:
        dataset = df[col].fillna(Vmediana,inplace=True)
        #print("Mediana Imputada")
        #print(Vmediana)
     elif list.index(mayor) == 2:
        dataset = df[col].fillna(Vmax,inplace=True)
        #print("Max Imputado")
        #print(Vmax)
     elif list.index(mayor) == 3:
        dataset = df[col].fillna(Varb,inplace=True)
        #print("Arb Imputado")
        #print(Varb)


    
    ### ============ Tratamiendo de Outliers ========
class OutliersTreatmentOperator(BaseEstimator, TransformerMixin):

    def __init__(self, factor = 1.75, varNames = None):
        self.varNames = varNames
        self.factor = factor

    def fit(self, X, y = None):
        for col in self.varNames:
            q3 = X[col].quantile(0.75)
            q1 = X[col].quantile(0.25)
            self.IQR = q3 - q1
            self.upper = q3 + self.factor*self.IQR
            self.lower = q1 - self.factor*self.IQR
        return self

    def transform(self, X, y = None):
        X = X.copy()
        for col in self.varNames:
            X[col] = np.where(X[col] >= self.upper, self.upper,
                np.where(
                    X[col] < self.lower, self.lower, X[col]
                )    
            )
        return X
    

def outlier_treatment(df, variable, factor):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    LI = df[variable].quantile(0.25) - (IQR*factor)
    LS = df[variable].quantile(0.75) + (IQR*factor)
    
    df[variable] = np.where(df[variable] > LS, LS, 
                                          np.where(df[variable] < LI, LI, df[variable]))