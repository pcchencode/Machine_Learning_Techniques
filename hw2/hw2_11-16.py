
# coding: utf-8

# In[1]:


## Date: 2018-05-29
## Purpose: Machine Learning Techniques hw2
## Author: Po-Chu Chen

#Q11~Q12
import numpy as np
import math

def lssvm(X, y,la,gamma):
    m, n = X.shape
    K = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            xi = X[i,:]
            xj = X[j,:]
            delta = xi-xj
            K[i,j] = math.exp(-gamma*np.sum(delta**2))
    beta = np.linalg.inv(la*np.eye(m)+K).dot(y)
    return beta, X          
        
def predict(beta, X, x, gamma):
    delta = X - x
    kf = np.exp(-gamma*np.sum(delta**2, axis=1))
    return beta.dot(kf)    
        
    
def error(beta, X, Xtest, y_test, gamma):
    y_predict_list = []
    for i in range(len(y_test)):
        x = Xtest[i,:]
        y_predict = predict(beta, X, x, gamma)
        y_predict_list.append(np.sign(y_predict))
    return np.sum(y_test!=np.array(y_predict_list)) / len(y_test)

data = np.loadtxt('hw2_lssvm_all.dat.txt')
X = data[:,:-1]
Y = data[:,-1]    
Xtrain = X[0:400, :]; Ytrain = Y[0:400]
Xtest = X[400:, :]; Ytest = Y[400:]

gammas = (32,2,0.125)
lamdas = (0.001,1,1000)
for gamma in gammas:
    for la in lamdas:
        beta, X = lssvm(Xtrain, Ytrain,la,gamma)
        e_in = error(beta,X,Xtrain,Ytrain,gamma)
        e_out = error(beta,X,Xtest,Ytest,gamma)
        print('gamma =', gamma,'lambda = ',la,'Ein = ',e_in,'Eout = ',e_out)


# In[235]:


## Q13~Q14 lssvm_revised
import numpy as np
import math


def lssvm(X,y,lamda):
    n, d = X.shape
    #K = np.zeros((m,m))
    #for i in range(m):
        #for j in range(m):
            #xi = X[i,:]
            #xj = X[j,:]
            #delta = xi-xj
            #K[i,j] = math.exp(-gamma*np.sum(delta**2))
    beta = np.linalg.inv((X.T.dot(X)+lamda*np.eye(d))).dot(X.T).dot(y)
    return beta       
        
def predict(beta, x):
    #delta = X - x
    #kf = np.exp(-gamma*np.sum(delta**2, axis=1))
    return beta.dot(x)    
        
    
def error(beta, Xtest, y_test):
    y_predict_list = []
    for i in range(len(y_test)):
        x = Xtest[i,:]
        y_predict = predict(beta, x)
        y_predict_list.append(np.sign(y_predict))
    return np.sum(y_test!=np.array(y_predict_list)) / len(y_test)

data = np.loadtxt('hw2_lssvm_all.dat.txt')
x0 = np.ones(data.shape[0])  # adding the constant feature
data=np.insert(data,0,x0,1)
#print(data)
X = data[:,:-1]
Y = data[:,-1]    
Xtrain = X[0:400, :]; Ytrain = Y[0:400]
Xtest = X[400:, :]; Ytest = Y[400:]


lamdas = (0.01,0.1,1,10,100)
#lamdas = (0.01,0.1)
for la in lamdas:
    beta = lssvm(Xtrain, Ytrain,la)
    e_in = error(beta,Xtrain,Ytrain)
    e_out = error(beta,Xtest,Ytest)
    print('lambda = ',la,'Ein = ',e_in,'Eout = ',e_out)


# In[232]:


#Q15~Q16 revised
import numpy as np
import math

data = np.loadtxt('hw2_lssvm_all.dat.txt')
x0 = np.ones(data.shape[0])  # adding the constant feature
data=np.insert(data,0,x0,1)
#print(data)
X = data[:,:-1]
Y = data[:,-1]
Xtrain = X[0:400, :]; Ytrain = Y[0:400]
Xtest = X[400:, :]; Ytest = Y[400:]

def lssvm(X,y,lamda):
    n, d = X.shape
    beta = np.linalg.inv((X.T.dot(X)+lamda*np.eye(d))).dot(X.T).dot(y)
    return beta       
        
def predict(beta, x):
    return beta.dot(x)  

T=250 #t should be 250
lamdas = (0.01,0.1,1,10,100)
beta_list=[] ; X_list=[]
for la in lamdas:
    for t in range(T): #t should be 250
        i = np.random.choice(400,400,replace=True)
        TrainBoot= data[i,:]
        Xtrainboot = TrainBoot[:,:-1]; Ytrainboot = TrainBoot[:,-1]
        beta = lssvm(Xtrainboot, Ytrainboot, la) # beta  是來自bootstrap data
        beta_list.append(beta) ;  #得到B1,B2,...,B250 and X1,X2,...,X250

    y_prediction_list=[]
    for i in range(len(Xtrain)):
        y_predict_list=[]
        for j in range(len(beta_list)):
            y_predict=np.sign(predict(beta_list[j],Xtrain[i]))
            y_predict_list.append(y_predict)
        if np.sum(y_predict_list) >0:
            y_prediction=1
        else:
            y_prediction=-1
        y_prediction_list.append(y_prediction)

    e_in=np.sum(Ytrain!=np.array(y_prediction_list)) / len(Ytrain)
    #print('Ein = ',e_in)

    y_prediction_list=[]
    for i in range(len(Xtest)):
        y_predict_list=[]
        for j in range(len(beta_list)):
            y_predict=np.sign(predict(beta_list[j], Xtest[i]))
            y_predict_list.append(y_predict)
        if np.sum(y_predict_list) >0:
            y_prediction=1
        else:
            y_prediction=-1
        y_prediction_list.append(y_prediction)

    e_out=np.sum(Ytest!=np.array(y_prediction_list)) / len(Ytest)
    #print('E_out = ',e_out)
    print('lambda = ',la,'Ein = ',e_in,'Eout = ',e_out)


# In[231]:


#Q15~Q16 kernel reidge regression with bagging(wrong_version)
import numpy as np
import math

data = np.loadtxt('hw2_lssvm_all.dat.txt')
x0 = np.ones(data.shape[0])  # adding the constant feature
data=np.insert(data,0,x0,1)
#print(data)
X = data[:,:-1]
Y = data[:,-1]
Xtrain = X[0:400, :]; Ytrain = Y[0:400]
Xtest = X[400:, :]; Ytest = Y[400:]

def lssvm(X,y,lamda):
    n, d = X.shape
    #K = np.zeros((m,m))
    #for i in range(m):
        #for j in range(m):
            #xi = X[i,:]
            #xj = X[j,:]
            #delta = xi-xj
            #K[i,j] = math.exp(-gamma*np.sum(delta**2))
    beta = np.linalg.inv((X.T.dot(X)+lamda*np.eye(d))).dot(X.T).dot(y)
    return beta       
        
def predict(beta, x):
    #delta = X - x
    #kf = np.exp(-gamma*np.sum(delta**2, axis=1))
    return beta.dot(x)  

T=250 #t should be 250
lamdas = (0.01,0.1,1,10,100)
beta_list=[] ; X_list=[]
for la in lamdas:
    for t in range(T): #t should be 250
        i = np.random.choice(400,400,replace=True)
        TrainBoot= data[i,:]
        Xtrainboot = TrainBoot[:,:-1]; Ytrainboot = TrainBoot[:,-1]
        beta = lssvm(Xtrainboot, Ytrainboot, la) # beta  是來自bootstrap data
        beta_list.append(beta) ;  #得到B1,B2,...,B250 and X1,X2,...,X250

    y_prediction_list=[]
    for i in range(len(Xtrain)):
        y_predict_list=[]
        for j in range(len(beta_list)):
            y_predict=np.sign(predict(beta_list[j],Xtrain[i]))
            y_predict_list.append(y_predict)
        if np.sum(y_predict_list) >0:
            y_prediction=1
        else:
            y_prediction=-1
        y_prediction_list.append(y_prediction)

    e_in=np.sum(Ytrain!=np.array(y_prediction_list)) / len(Ytrain)
    #print('Ein = ',e_in)

    y_prediction_list=[]
    for i in range(len(Xtest)):
        y_predict_list=[]
        for j in range(len(beta_list)):
            y_predict=np.sign(predict(beta_list[j], Xtest[i]))
            y_predict_list.append(y_predict)
        if np.sum(y_predict_list) >0:
            y_prediction=1
        else:
            y_prediction=-1
        y_prediction_list.append(y_prediction)

    e_out=np.sum(Ytest!=np.array(y_prediction_list)) / len(Ytest)
    #print('E_out = ',e_out)
    print('lambda = ',la,'Ein = ',e_in,'Eout = ',e_out)









