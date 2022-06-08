import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import xgboost as xgb
import time

seed=47

invoice_test = pd.read_csv(r"C:\Users\pc gamer casa\Desktop\invoice_test.csv",low_memory=False)
invoice_train = pd.read_csv(r"C:\Users\pc gamer casa\Desktop\invoice_train.csv",low_memory=False)
client_test = pd.read_csv(r"C:\Users\pc gamer casa\Desktop\client_test.csv",low_memory=False)
client_train = pd.read_csv(r"C:\Users\pc gamer casa\Desktop\client_train.csv",low_memory=False)
sample_submission = pd.read_csv(r"C:\Users\pc gamer casa\Desktop\SampleSubmission.csv",low_memory=False)


ds = client_train.groupby(['target'])['client_id'].count()
plt.bar(x=ds.index, height=ds.values, tick_label =[0,1])
plt.title('target distribution')
#plt.show()

#district
for col in ['disrict']:
    ds = client_train.groupby([col])['client_id'].count()
    plt.bar(x=ds.index, height=ds.values)
    plt.title(col+' distribution')
    #plt.show()
    
    
 #region   
for col in ['region']:
    ds = client_train.groupby([col])['client_id'].count()
    plt.bar(x=ds.index, height=ds.values)
    plt.title(col+' region')
   # plt.show()
    
    
#clientCategory
for col in ['client_catg']:
    ds = client_train.groupby([col])['client_id'].count()
    plt.bar(x=ds.index, height=ds.values)
    plt.title(col+' client catg')
   # plt.show()
    
    
    
print('Number of missing rows in invoice_train:',invoice_train.isna().sum().sum())
print('Number of missing rows in invoice_test:',invoice_test.isna().sum().sum(),'\n')
print('Number of missing rows in client_train:',client_train.isna().sum().sum())
print('Number of missing rows in client_test:',client_test.isna().sum().sum())


print('Number of unique values in invoice_train:')
for col in invoice_train.columns:
    print(f"{col} - {invoice_train[col].nunique()}")
    
    
    
    
def feature_change(cl, inv):
    cl['region_group'] = cl['region'].apply(lambda x: 100 if x<100 else 300 if x>300 else 200)
    cl['creation_date'] = pd.to_datetime(cl['creation_date'])
    
    cl['coop_time'] = (2019 - cl['creation_date'].dt.year)*12 - cl['creation_date'].dt.month

    inv['counter_type'] = inv['counter_type'].map({"ELEC":1,"GAZ":0})
    inv['counter_statue'] = inv['counter_statue'].map({0:0,1:1,2:2,3:3,4:4,5:5,769:5,'0':0,'5':5,'1':1,'4':4,'A':0,618:5,269375:5,46:5,420:5})
    
    inv['invoice_date'] = pd.to_datetime(inv['invoice_date'], dayfirst=True)
    inv['invoice_month'] = inv['invoice_date'].dt.month
    inv['invoice_year'] = inv['invoice_date'].dt.year
    inv['is_weekday'] = ((pd.DatetimeIndex(inv.invoice_date).dayofweek) // 5 == 1).astype(float)
    inv['delta_index'] = inv['new_index'] - inv['old_index']
    
    return cl, inv


client_train1, invoice_train1 = feature_change(client_train, invoice_train)
client_test1, invoice_test1 = feature_change(client_test, invoice_test)


def agg_feature(invoice, client_df, agg_stat):
    
    invoice['delta_time'] = invoice.sort_values(['client_id','invoice_date']).groupby('client_id')['invoice_date'].diff().dt.days.reset_index(drop=True)
    agg_trans = invoice.groupby('client_id')[agg_stat+['delta_time']].agg(['mean','std','min','max'])
    
    agg_trans.columns = ['_'.join(col).strip() for col in agg_trans.columns.values]
    agg_trans.reset_index(inplace=True)

    df = invoice.groupby('client_id').size().reset_index(name='transactions_count')
    agg_trans = pd.merge(df, agg_trans, on='client_id', how='left')
    
    weekday_avg = invoice.groupby('client_id')[['is_weekday']].agg(['mean'])
    weekday_avg.columns = ['_'.join(col).strip() for col in weekday_avg.columns.values]
    weekday_avg.reset_index(inplace=True)
    client_df = pd.merge(client_df, weekday_avg, on='client_id', how='left')
    
    full_df = pd.merge(client_df, agg_trans, on='client_id', how='left')
    
    full_df['invoice_per_cooperation'] = full_df['transactions_count'] / full_df['coop_time']
    
    return full_df


agg_stat_columns = [
 'tarif_type',
 'counter_number',
 'counter_statue',
 'counter_code',
 'reading_remarque',
 'consommation_level_1',
 'consommation_level_2',
 'consommation_level_3',
 'consommation_level_4',
 'old_index',
 'new_index',
 'months_number',
 'counter_type',
 'invoice_month',
 'invoice_year',
 'delta_index'
]

train_df1 = agg_feature(invoice_train1, client_train1, agg_stat_columns)
test_df1 = agg_feature(invoice_test1, client_test1, agg_stat_columns)


def drop(df):

    col_drop = ['client_id', 'creation_date']
    for col in col_drop:
        df.drop([col], axis=1, inplace=True)
    return df

# Function to calculate accuracy
def cal_accuracy(y_test, y_pred):
      
    print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)
      
    print("Report : ",
    classification_report(y_test, y_pred))



train_df = drop(train_df1)
test_df = drop(test_df1)


df=train_df.replace((np.inf, -np.inf, np.nan), 0).reset_index()
test_df=test_df.replace((np.inf, -np.inf, np.nan), 0).reset_index()


y = df['target']
X = df.drop('target',axis=1)

print(X.corr())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



over_sampler = RandomOverSampler(random_state=42)
X_res, y_res = over_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")


def plotgraph(test,prediction,i,name):
    confusion_matrix(test, prediction) 
    fpr, tpr, threshold = metrics.roc_curve(test, prediction)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(i)
    plt.plot(fpr, tpr, 'g', label = 'AUC = %0.2f' % roc_auc)
    plt.title(name)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


#Logistic regression
regr = LogisticRegression()
regr.fit(X_res, y_res)
Y_pred=regr.predict(X_test)
score1=regr.score(X_test,y_test)
mse1=mean_squared_error(Y_pred,y_test)
plotgraph(y_test, Y_pred, 4,"Logistic regression")
print("logistic regression")
cal_accuracy(y_test,Y_pred)

#naive bayes
model=GaussianNB()
model.fit(X_res, y_res)
Y_pred1=model.predict(X_test)
Score2=model.score(X_test,y_test)
mse2=mean_squared_error(Y_pred,y_test)
plotgraph(y_test, Y_pred1, 5,"Naive bayes")
print("naive bayes")
cal_accuracy(y_test,Y_pred1)

#testing lda
clf = LinearDiscriminantAnalysis()
clf.fit(X_res, y_res)
Y_pred2=clf.predict(X_test)
Score3=clf.score(X_test,y_test)
mse3=mean_squared_error(Y_pred,y_test)
plotgraph(y_test, Y_pred2, 6,"LDA")
print("LDA")
cal_accuracy(y_test,Y_pred2)

#### QDA
clf = QuadraticDiscriminantAnalysis()
clf.fit(X_res, y_res)
Y_pred3=clf.predict(X_test)
Score4=clf.score(X_test,y_test)
plotgraph(y_test, Y_pred3, 4,"QDA")
print("QDA")
cal_accuracy(y_test,Y_pred3)

#KNN
#neigh = KNeighborsClassifier(n_neighbors=12)
#neigh.fit(X_train,y_train) 
#Y_pred3=neigh.predict(X_test)
#Y_pred31=neigh.predict_proba(X_test)
#Score5=neigh.score(X_test,y_test)
#plotgraph(y_test, Y_pred3, 5,"KNN")
#print("logistic regression")
#cal_accuracy(y_test,Y_pred3)


#decision tree
def train_using_gini(X_train, X_test, y_train):
  
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 100,max_depth=3, min_samples_leaf=5)
  
    # Performing training
    clf_gini.fit(X_train, y_train)
    return clf_gini
      
# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
  
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier(
    criterion = "entropy", random_state = 100,
    max_depth = 3, min_samples_leaf = 5)
  
    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy
  
  
# Function to make predictions
def prediction(X_test, clf_object):
  
    # Predicton on test with giniIndex
    y_pred = clf_object.predict(X_test)
    print("Predicted values:")
    print(y_pred)
    return y_pred
      






clf_gini = train_using_gini(X_train, X_test, y_train)
clf_entropy = tarin_using_entropy(X_train, X_test, y_train)
      
    # Operational Phase
print("Results Using Gini Index:")
      
    # Prediction using gini
y_pred_gini = prediction(X_test, clf_gini)
cal_accuracy(y_test, y_pred_gini)
  
plotgraph(y_test, y_pred_gini, 6, "gini ")    
print("Results Using Entropy:")
    # Prediction using entropy
y_pred_entropy = prediction(X_test, clf_entropy)
cal_accuracy(y_test, y_pred_entropy)
plotgraph(y_test, y_pred_entropy, 7, "Entropy")

















#boosting xgc

dtrain = xgb.DMatrix(X_train,label=y_train)
dtest= xgb.DMatrix(X_test)

param = {'max_depth': 1, 'eta': 0.5, 'objective': 'binary:logistic'}
param['nthread'] = 11
param['eval_metric'] = 'auc'

evallist = [(dtest, 'eval'), (dtrain, 'train')]




num_round = 30
bst = xgb.train(param, dtrain, num_round, evallist)





def predictboost(test,model):
        ntest= xgb.DMatrix(test)
        probs=model.predict(ntest)
        
        return probs
    
def predictnormal(test,model):
    probs=model.predict(test)
    return probs


probs=predictboost(test_df, bst)
submission = pd.DataFrame({
        "client_id": sample_submission["client_id"],
        "target": probs.tolist()
    })
submission.to_csv('submission.csv', index=False)






