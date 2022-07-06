#!/usr/bin/env python
# coding: utf-8

# In[13]:


import modulosP2 as mod
import preprocessor as pp
from select import select
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import os
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import time
from datetime import datetime

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

inp = input("Presione Enter para realizar Pipeline y Entrenamiento del modelo ...")

def main():
    
    #def Train_Model(dataset):
        
        data = pd.read_csv('DataProyecto.csv', sep = ",", encoding='latin-1')
        dataset = data

        dataset = dataset.drop(['Precio', 'Cantidad', 'Almacen_Recurrente', 'Vehiculo', 'Proyecto' ,  'Descuento_Lineal' , 'Moneda', 'Tipo_Cambio' , 'Sub_Contrato','Almacen_CT' , 'Tipo_Periodo', 'Grupo' , 'Transaccion_Ventas' , 'Pais' ,  'Importe' , 'Usuario', 'Agente','Impuestos' ,'Familia','Movimiento_Ventas'], axis=1)
        
        mod.getNumColNames(dataset)

        mod.getCatColNames(dataset)

        cols_num_con_nan = mod.getNumNanColNames(dataset)

        for col in cols_num_con_nan:
                   pp.imputVN(dataset, col ,"Tipo_Contrato"),

        cols_cca = mod.getNanGoodColsNames(dataset, 0.05)

        dataset_temp = dataset[cols_cca].dropna()

        dataset_vn = dataset_temp

        cols_cat_con_nan = mod.getCatNanColNames(dataset_vn)
        
        for col in cols_cat_con_nan:
            pp.ImpVC(dataset_vn)
            
        dataset_vc = dataset_vn
        
        cat_cols = mod.getCategoryVars(dataset_vc)

        for col in cat_cols:
            pp.executeFreqEncoding(dataset_vc, map_type='freq')
            
        dataset_cvc = dataset_vc
        
        numeric_cont_vars = mod.getContinuesCols(dataset_cvc)

        #for col in numeric_cont_vars:
            #mod.plot_density_variable(dataset_cvc, col)
        
        for col in numeric_cont_vars:
            pp.transfvnum(dataset_cvc, col,"Tipo_Contrato")
            
        dataset_tvn = dataset_cvc

        #for col in numeric_cont_vars:
         #    mod.inspect_outliers(dataset_tvn, col)
            
        for col in numeric_cont_vars:
            mod.detect_outliers(dataset_tvn, col, 1.75)
        
        for col in numeric_cont_vars:
            pp.outlier_treatment(dataset_tvn, col, 1.75)
            
        dataset_sca = dataset_tvn
        
        dataset_sca[dataset_sca.columns[1:34]]
        
        scaler = StandardScaler()
        scaler.fit(dataset_sca) 
        
        dataset_final = pd.DataFrame(scaler.transform(dataset_sca), columns=dataset_sca.columns)
        
        np.round(dataset_final.describe(), 2)
        
        dataset_final['Tipo_Contrato'] = dataset_tvn['Tipo_Contrato'].values
        
        dataset_final.loc[dataset_final.Tipo_Contrato == 1020, "Tipo_Contrato"] = 1
        dataset_final.loc[dataset_final.Tipo_Contrato == 1091, "Tipo_Contrato"] = 0
        
        dataset_final.to_csv("DataTratada.csv", encoding = 'latin-1')
        
        dataset = pd.read_csv('DataTratada.csv', encoding = 'latin-1')
        dataset = dataset.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
        
        X = dataset.drop(['Tipo_Contrato'], axis=1)
        y = dataset['Tipo_Contrato']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=2022)
        
        inicio = time.time()
        Nb = GridSearchCV(GaussianNB(), {'var_smoothing': [0.9]},cv=10).fit(X_train, y_train)
        Log = GridSearchCV(LogisticRegression(), {'solver':['liblinear'], 'verbose':[3],'max_iter':[100]},  cv=10).fit(X_train, y_train)
        LDA = GridSearchCV(LinearDiscriminantAnalysis(), {'solver':['svd','lsqr']},  cv=10).fit(X_train, y_train)
        svm = GridSearchCV(SVC(), {'C': [0.1], 'kernel': ['linear']}, cv=10).fit(X_train, y_train)
        randFor = GridSearchCV(RandomForestClassifier(), {'n_estimators': [20, 50, 100], 'max_depth': [10, 100, 200]}).fit(X_train, y_train)
        AB = GridSearchCV(AdaBoostClassifier(),{'n_estimators': [55]},cv=10).fit(X_train, y_train)
        #GB = GridSearchCV(GradientBoostingClassifier(), {'loss':['log_loss']},cv=10).fit(X_train, y_train)
        TD = GridSearchCV(DecisionTreeClassifier(), {'criterion':['gini'], 'min_impurity_decrease':[0.000001], 'random_state':[2]},cv=10).fit(X_train, y_train)
        QuadDA = GridSearchCV(QuadraticDiscriminantAnalysis(), {'store_covariance': ['T']},cv=10).fit(X_train, y_train)
        LinearDA = GridSearchCV(LinearDiscriminantAnalysis(), {'solver': ['lsqr']},cv=2).fit(X_train, y_train)
        xgb = XGBClassifier(objective="binary:logistic", random_state=42)
        xgb.fit(X_train, y_train)
        LGBM = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)
        LGBM.fit(X_train,y_train,eval_set=[(X_test,y_test),(X_train,y_train)],
                  verbose=20,eval_metric='logloss')
        fin = time.time()
        
        y_preds_NB = Nb.predict(X_test)
        y_preds_Log = Log.predict(X_test)
        y_preds_LDA = LDA.predict(X_test)
        y_preds_svm = svm.predict(X_test)
        y_preds_randFor = randFor.predict(X_test)
        y_preds_QuadDA = QuadDA.predict(X_test)
        y_preds_LinearDA = LinearDA.predict(X_test)
        y_preds_AB = AB.predict(X_test)
        y_preds_TD = TD.predict(X_test)
        y_preds_LGBM = LGBM.predict(X_test)
        y_preds_XGBoost = xgb.predict(X_test) 
       
        conf_matrixNB = pd.crosstab(y_test, y_preds_NB, rownames=["observación"], colnames=["Predicción"])
        conf_matrixLDA = pd.crosstab(y_test, y_preds_LDA, rownames=["observación"], colnames=["Predicción"])
        conf_matrixLog = pd.crosstab(y_test, y_preds_Log, rownames=["observación"], colnames=["Predicción"])
        conf_matrixSVM = pd.crosstab(y_test, y_preds_svm, rownames=["observación"], colnames=["Predicción"])
        conf_matrixrandFor = pd.crosstab(y_test, y_preds_randFor, rownames=["observación"], colnames=["Predicción"])
        conf_matrixQuadDA = pd.crosstab(y_test, y_preds_QuadDA, rownames=["observación"], colnames=["Predicción"])
        conf_matrixLinearDA = pd.crosstab(y_test, y_preds_LinearDA, rownames=["observación"], colnames=["Predicción"])
        conf_matrixAB = pd.crosstab(y_test, y_preds_AB, rownames=["observación"], colnames=["Predicción"])
        conf_matrixTD = pd.crosstab(y_test, y_preds_TD, rownames=["observación"], colnames=["Predicción"])
        conf_matrixLGBM = pd.crosstab(y_test, y_preds_LGBM, rownames=["observación"], colnames=["Predicción"])
        conf_matrixXGBoost = pd.crosstab(y_test, y_preds_XGBoost, rownames=["observación"], colnames=["Predicción"])
      
             
        list = [roc_auc_score(y_test, y_preds_NB, multi_class='ovr'), roc_auc_score(y_test, y_preds_Log, multi_class='ovo'),roc_auc_score(y_test, y_preds_LDA, multi_class='ovo'),roc_auc_score(y_test, y_preds_TD, multi_class='ovo'),roc_auc_score(y_test, y_preds_svm, multi_class='ovo'),roc_auc_score(y_test, y_preds_randFor, multi_class='ovo'),roc_auc_score(y_test, y_preds_QuadDA, multi_class='ovo'),roc_auc_score(y_test, y_preds_LinearDA, multi_class='ovo'),roc_auc_score(y_test, y_preds_AB, multi_class='ovo'),roc_auc_score(y_test, y_preds_LGBM, multi_class='ovo'),roc_auc_score(y_test, y_preds_XGBoost, multi_class='ovo')]
        mejor = max(list)
        pos = list.index(mejor)

        if list.index(mejor) == 0:
            print("El mejor moodelo es Naive Bayes")
            TP = conf_matrixNB.iloc[1,1]
            TN = conf_matrixNB.iloc[0,0]
            FN = conf_matrixNB.iloc[1,0]
            FP = conf_matrixNB.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_NB, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_NB)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Naive Bayes" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Naive Bayes = " +str(roc_auc_score(y_test, y_preds_NB, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 1:
            print("El mejor moodelo es Logistica")
            TP = conf_matrixLog.iloc[1,1]
            TN = conf_matrixLog.iloc[0,0]
            FN = conf_matrixLog.iloc[1,0]
            FP = conf_matrixLog.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_Log, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_Log)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Logistica" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Logistica = " +str(roc_auc_score(y_test, y_preds_Log, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()


        elif list.index(mejor) == 2:
            print("El mejor moodelo es LDA")
            TP = conf_matrixLDA.iloc[1,1]
            TN = conf_matrixLDA.iloc[0,0]
            FN = conf_matrixLDA.iloc[1,0]
            FP = conf_matrixLDA.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_LDA, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_LDA)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es LDA" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ TN/(TN+FP) + os.linesep)
            file.write("ROC-AUC LDA = " +str(roc_auc_score(y_test, y_preds_LDA, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 3:
            print("El mejor moodelo es Tree Decision")
            TP = conf_matrixTD.iloc[1,1]
            TN = conf_matrixTD.iloc[0,0]
            FN = conf_matrixTD.iloc[1,0]
            FP = conf_matrixTD.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_TD, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_TD)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Tree Decision" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Tree Decision = " +str(roc_auc_score(y_test, y_preds_TD, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 4:
            print("El mejor moodelo es SVM")
            TP = conf_matrixSVM.iloc[1,1]
            TN = conf_matrixSVM.iloc[0,0]
            FN = conf_matrixSVM.iloc[1,0]
            FP = conf_matrixSVM.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precisio: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_SVM, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio))
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_SVM)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es SVM" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC SVM = " +str(roc_auc_score(y_test, y_preds_SVM, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 5:
            print("El mejor moodelo es  Random Forest")
            TP = conf_matrixrandFor.iloc[1,1]
            TN = conf_matrixrandFor.iloc[0,0]
            FN = conf_matrixrandFor.iloc[1,0]
            FP = conf_matrixrandFor.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_randFor, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio))
            print('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_randFor)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Random Forest" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Random Forest = " +str(roc_auc_score(y_test, y_preds_randFor, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 6:
            print("El mejor moodelo es  Quadratic Discriminant Analisys")
            TP = conf_matrixQuadDA.iloc[1,1]
            TN = conf_matrixQuadDA.iloc[0,0]
            FN = conf_matrixQuadDA.iloc[1,0]
            FP = conf_matrixQuadDA.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_QuadDA, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_QuadDA)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Quadratic Discriminant Analisys" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Quadratic Discriminant Analisys = " +str(roc_auc_score(y_test, y_preds_y_preds_QuadDA, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 7:
            print("El mejor moodelo es  Linear Discriminant Analisys")
            TP = conf_matrixLinearDA.iloc[1,1]
            TN = conf_matrixLinearDA.iloc[0,0]
            FN = conf_matrixLinearDA.iloc[1,0]
            FP = conf_matrixLinearDA.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_LinearDA, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_LinearDA)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Linear Discriminant Analisys" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Linear Discriminant Analisys = " +str(roc_auc_score(y_test, y_preds_y_preds_LinearDA, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+ str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 8:
            print("El mejor moodelo es  Ada Boost")
            TP = conf_matrixAB.iloc[1,1]
            TN = conf_matrixAB.iloc[0,0]
            FN = conf_matrixAB.iloc[1,0]
            FP = conf_matrixAB.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_AB, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_AB)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es Ada Boost" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC Ada Boost = " +str(roc_auc_score(y_test, y_preds_y_preds_AB, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+ str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 9:
            print("El mejor moodelo es  LGBM")
            TP = conf_matrixLGBM.iloc[1,1]
            TN = conf_matrixLGBM.iloc[0,0]
            FN = conf_matrixLGBM.iloc[1,0]
            FP = conf_matrixLGBM.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_LGBM, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: ',fin-inicio)
            print('Fecha y hora de ejecución: ',datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_LGBM)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es LGBM" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC LGBM = " +str(roc_auc_score(y_test, y_preds_LGBM, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+ str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close()

        elif list.index(mejor) == 10:
            print("El mejor moodelo es  XGBoost")
            TP = conf_matrixXGBoost.iloc[1,1]
            TN = conf_matrixXGBoost.iloc[0,0]
            FN = conf_matrixXGBoost.iloc[1,0]
            FP = conf_matrixXGBoost.iloc[0,1]
            print("Exactitud: ", (TP+TN)/(TP+TP+TN+TN))
            print("Precision: ", TP/(TP+TP) )
            print("Sensitividad: ", TP/(TP+FN))
            print("Especificidad: ", TN/(TN+FP))
            print("ROC AUC: ",roc_auc_score(y_test, y_preds_XGBoost, multi_class='ovr'))
            print('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio))
            print('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            df = pd.DataFrame(y_preds_XGBoost)
            df.to_csv("Predicciones.csv", encoding = 'UTF-8')
            file = open("Ejecucion.txt", "w")
            file.write("El mejor moodelo es XGBoost" + os.linesep)
            file.write("Exactitud: "+ str((TP+TN)/(TP+TP+TN+TN)) + os.linesep)
            file.write("Precision: "+ str(TP/(TP+TP)) + os.linesep)
            file.write("Sensitividad: "+ str(TP/(TP+FN)) + os.linesep)
            file.write("Especificidad: "+ str(TN/(TN+FP)) + os.linesep)
            file.write("ROC-AUC XGBoost = " +str(roc_auc_score(y_test, y_preds_XGBoost, multi_class='ovo')) + os.linesep)
            file.write('El tiempo de entrenamiento de modelos tardo: '+str(fin-inicio) + os.linesep)
            file.write('Fecha y hora de ejecución: '+datetime.today().strftime('%Y-%m-%d %H:%M'))
            file.close() 
 
#inp = input("Presione Enter para generar Predicción ...")

main()


# In[ ]:




