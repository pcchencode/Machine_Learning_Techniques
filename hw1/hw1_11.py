## Date: 2018-04-21
## Purpose: Machine Learning Techniques hw1
## Author: Po-Chu Chen

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def load_data(fname):
    data = np.loadtxt(fname)
    X = data[:,1:]
    y = data[:,0]
    return X, y

X, y = load_data('features.train.txt')
X_test, y_test = load_data('features.test.txt')



## Problem 11.
y0 = np.where(y==0, 1 ,-1)
all_dw = []
logCs = (-5,-3,-1,1,3) # such a long time..zzz
#logCs = (-5,-3,-1) # reduce the running time for trial
for logC in logCs:
    C = 10**logC
    clf = svm.SVC(C=C, kernel='linear',shrinking=False)    
    clf.fit(X ,y0)
    w = clf.coef_
    dw = np.linalg.norm(w)
    all_dw.append(dw)
    
print('#11:', all_dw)
plt.figure()
plt.plot(logCs, all_dw)
plt.xlabel("logCs")
plt.ylabel("||w||")
plt.show()


## Problem 12. 13.
all_e_in=[]
all_num_sv=[]
y8 = np.where(y==8, 1 ,-1)
logCs = (-5,-3,-1, 1, 3) # such a long time...
#logCs = (-5,-3,-1) # reduce the running time for trial

# since the package does not contain the error function, we define it by ourselves
def error(y_preditc, y):
    return np.sum(y_preditc!=y) / len(y);

for logC in logCs:
    C = 10**logC
    poly_clf = svm.SVC(C=C, kernel='poly', degree=2, gamma=1, coef0=1, shrinking=False)
    poly_clf.fit(X, y8)
    y_predict = poly_clf.predict(X)
    e_in = error(y_predict, y8)
    num_sv = np.count_nonzero(np.abs(poly_clf.dual_coef_)) # dual_coef_ = alpha * y
    all_e_in.append(e_in)
    all_num_sv.append(num_sv)

print('#12:', all_e_in)
plt.figure()
plt.plot(logCs, all_e_in)
plt.xlabel("logCs")
plt.ylabel("Ein")
plt.show()

print('#13:', all_num_sv)
plt.figure()
plt.plot(logCs, all_num_sv)
plt.xlabel("logCs")
plt.ylabel("# of SV")
plt.show()



## Problem 14.
all_d = []
logCs = (-3, -2,-1,0,1)
for logC in logCs:
    C = 10**logC
    rbf_clf = svm.SVC(C=C, kernel='rbf',shrinking=False, gamma=80)    
    rbf_clf.fit(X ,y0)
    alphas = np.abs(rbf_clf.dual_coef_)    
    for i in range(rbf_clf.dual_coef_.size):
        alpha = alphas[0,i]
        if alpha<C and alpha > 0: # condition for free support vector
            x = rbf_clf.support_vectors_[i].reshape(1, -1)
            decision_value = rbf_clf.decision_function(x)[0]
            # distance = |decision_value| / |w|
            w = clf.coef_
            dw = np.linalg.norm(w)
            b = clf.intercept_
            distance = np.abs(decision_value) / (dw)
            all_d.append(distance)
            break
print('#14', all_d)
plt.figure()
plt.plot(logCs, all_d)
plt.xlabel("logCs")
plt.ylabel("distance")
plt.show()



## Problem 15
all_e_out = []
y0_test = np.where(y_test==0, 1 ,-1)
log_gammas = (0,1,2,3,4)
#log_gammas = (0,1)
for log_gamma in log_gammas:
    gamma = 10 ** log_gamma
    rbf_clf = svm.SVC(C=0.1, kernel='rbf',shrinking=False, gamma=gamma)    
    rbf_clf.fit(X ,y0)
    y0_test_predict = rbf_clf.predict(X_test) # Eout is for the test data
    e_out = error(y0_test_predict, y0_test)
    all_e_out.append(e_out)
print('#15:', all_e_out)
plt.figure()
plt.plot(log_gammas, all_e_out)
plt.xlabel("logGamma")
plt.ylabel("Eout")
plt.show()
# at gamma=1, Eout reach the smallest value


## Problem 16.
all_e_val =[]
log_gammas = (-1,0,1,2,3)
for log_gamma in log_gammas:
    gamma = 10 ** log_gamma
    rbf_clf = svm.SVC(C=0.1, kernel='rbf',shrinking=False, gamma=gamma)
    errors=[]
    for i in range(100):
        indexs = np.random.permutation(y.size)
        X_val = X[indexs[0:1000],:]
        y_val = y0[indexs[0:1000]]
        X_train = X[indexs[1000:],:]
        y_train = y0[indexs[1000:]]
        rbf_clf.fit(X_train, y_train)
        y_predict =rbf_clf.predict(X_val)
        errors.append(error(y_predict, y_val))
    all_e_val.append(np.mean(errors))
print('#16:', all_e_val)
plt.figure()
plt.plot(log_gammas, all_e_val)
plt.xlabel("logGamma")
plt.ylabel("Eval")
plt.show()
# the gamma for smallest Eval is gamma=1, which is same as the previous problem



## Another way to compute # of support vector(problem 13)
all_e_in=[]
all_num_sv=[]
y8 = np.where(y==8, 1 ,-1)
logCs = (-5,-3,-1, 1, 3) # such a long time...
#logCs = (-5,-3,-1) # reduce the running time for trial

# since the package does not contain the error function, we define it by ourselves
def error(y_preditc, y):
    return np.sum(y_preditc!=y) / len(y);

for logC in logCs:
    C = 10**logC
    poly_clf = svm.SVC(C=C, kernel='poly', degree=2, gamma=1, coef0=1, shrinking=False)
    poly_clf.fit(X, y8)
    y_predict = poly_clf.predict(X)
    e_in = error(y_predict, y8)
    #num_sv = np.count_nonzero(np.abs(poly_clf.dual_coef_)) #dual_coef_ = alpha * y
    num_sv = np.sum(np.abs(poly_clf.n_support_)) # package can derive the # of SV directly
    all_num_sv.append(num_sv)


print('#13:', all_num_sv)
plt.figure()
plt.plot(logCs, all_num_sv)
plt.xlabel("logCs")
plt.ylabel("# of SV")
plt.show()

