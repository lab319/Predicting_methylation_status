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
from sklearn.ensemble import RandomForestClassifier
batch_size = 32
nb_classes = 10
nb_epoch = 10
def RF(traindata,trainlabel,testdata,testlabel):
    print("start training RF")
    clf=RandomForestClassifier(n_estimators=1000)
    #y_pred=clf.predict()
    clf.fit(traindata,trainlabel)
    y_pred=clf.predict(testdata)
    cm=confusion_matrix(testlabel,y_pred)
    tn, fp, fn, tp = cm.ravel()
    Y_prob=clf.predict_proba(testdata)
    #Y_pred=clf.predict_proba(testdata)
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
def RFF(traindata,trainlabel,testdata,testlabel):
    print("start training RF")
    clf=RandomForestClassifier(n_estimators=1000)
    #y_pred=clf.predict()
    clf.fit(traindata,trainlabel)
    y_pred=clf.predict(testdata)
    cm=confusion_matrix(testlabel,y_pred)
    tn, fp, fn, tp = cm.ravel()
    Y_prob=clf.predict_proba(testdata)
    #Y_pred=clf.predict_proba(testdata)
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
matrix=np.loadtxt(".data\\DNA methylation feature.csv")
label=np.loadtxt(".data\\DNA methylation status.csv")
train=StandardScaler().fit_transform(matrix)
X_train=train[0:300000,]
X_test=train[300000:378677,]
Y_test=label[300000:378677,]
Y_train=label[0:300000,]
model = Sequential()
print("The first fully connected layer")
#model.add(Dense(output_dim=20,input_dim=122,init='uniform',W_regularizer=l2(0.01)))
model.add(Dense(output_dim=51,input_dim=122))
#model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
model.add(Activation('tanh'))
print("The second fully connected layer")
model.add(Dense(output_dim=26))
model.add(Activation('tanh'))
print("The third fully connected layer")
model.add(Dense(output_dim=10))
model.add(Activation('tanh'))
print("The fourth fully connected layer")
model.add(Dense(output_dim=5))
model.add(Activation('tanh'))
model.add(Dense(output_dim=3))
model.add(Activation('tanh'))
model.add(Dense(1,init='uniform'))
model.add(Activation('sigmoid'))
sgd = SGD(lr=0.01,decay=1e-5, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
hist=model.fit(X_train, Y_train, batch_size=16, nb_epoch=9,verbose=1,validation_split=0.1,shuffle=False)
cPickle.dump(model,open(".\\model.pkl","wb"))
###############Import DNN model#########################
origin_model = cPickle.load(open(".\\model.pkl","rb"))
y_pred=origin_model.predict_classes(X_test)
Y_pred=origin_model.predict(X_test)
cm=confusion_matrix(Y_test,y_pred)
ACC=accuracy_score(Y_test,y_pred)
AUC=roc_auc_score(Y_test,Y_pred)
print("DNN MCC: %f " %matthews_corrcoef(Y_test,y_pred))
print( "DNN ACC:  %f "  %accuracy_score(Y_test,y_pred))
print( "DNN AUC:  %f "  %roc_auc_score(Y_test,Y_pred))
get_feature=K.function([origin_model.layers[0].input,K.learning_phase()],origin_model.layers[9].output)
FC_train_feature = get_feature([X_train.astype('float32'),1])
FC_test_feature = get_feature([X_test.astype('float32'),0])
X_trainnn=np.concatenate((X_train,FC_train_feature),axis=1)
X_testtt=np.concatenate((X_test,FC_test_feature),axis=1)
RFF(FC_train_feature,Y_train,FC_test_feature,Y_test)
RF(X_trainnn,Y_train,X_testtt,Y_test)