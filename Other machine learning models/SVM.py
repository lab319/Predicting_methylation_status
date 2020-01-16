from __future__ import print_function
import numpy as np
np.random.seed(100)  # for reproducibility
import numpy as np
import sklearn
import theano
import random,cPickle
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Reshape
from keras.optimizers import SGD, Adadelta, Adagrad
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import  roc_auc_score
from sklearn.metrics import  accuracy_score
from keras import backend as K
from sklearn import svm
matrix=np.loadtxt(".data\\DNA methylation feature.csv")
label=np.loadtxt(".data\\DNA methylation status.csv")
train=StandardScaler().fit_transform(matrix)
X_train=train[0:300000,]
X_test=train[300000:378677,]
Y_test=label[300000:378677,]
Y_train=label[0:300000,]
def SVM(traindata,trainlabel,testdata,testlabel):
    print("start training SVM")
    clf= svm.SVC(C=2**5,kernel="rbf",gamma=2**(-7),cache_size=3000,probability=True)
    clf.fit(traindata,trainlabel)
    y_pred=clf.predict(testdata)
    cm=confusion_matrix(testlabel,y_pred)
    tn, fp, fn, tp = cm.ravel()
    Y_prob=clf.predict_proba(testdata)
    ACC=accuracy_score(testlabel,y_pred)
    AUC=roc_auc_score(testlabel,Y_prob[:,1])
    SE=float((tp)/(tp+fn))
    SP=float((tn)/(tn+fp))
    print("SE: %f " %SE)
    print("SP: %f " %SP)
    print("MCC: %f " %matthews_corrcoef(testlabel,y_pred))
    print( "ACC:  %f "  %accuracy_score(testlabel,y_pred))
    print( "AUC:  %f "  %roc_auc_score(testlabel,y_pred))
    #print("probability AUC:%f"%roc_auc_score(testlabel,Y_pred[:,0]))
    print("probability AUC:%f" %roc_auc_score(testlabel,Y_prob[:,1]))
    print(Y_prob)
    print(cm)
origin_model = cPickle.load(open(".\\model.pkl","rb"))
get_feature=K.function([origin_model.layers[0].input,K.learning_phase()],origin_model.layers[9].output)
FC_train_feature = get_feature([X_train.astype('float32'),1])
FC_test_feature = get_feature([X_test.astype('float32'),0])
X_trainnn=np.concatenate((X_train,FC_train_feature),axis=1)
X_testtt=np.concatenate((X_test,FC_test_feature),axis=1)
SVM(X_trainnn,Y_train,X_testtt,Y_test)
