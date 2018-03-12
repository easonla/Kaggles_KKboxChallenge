import time
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from xgboost import plot_importance

print ("Loading Data")
train_data = pd.read_csv('training_feature_v2.csv')
test_data  = pd.read_csv('testing_feature_v2.csv')
catagory = ['payment_method_id', 'payment_plan_days', 'plan_list_price','actual_amount_paid', 'city','gender', 'registered_via']
encoder = {}
for c in catagory:
    print (c)
    le = LabelEncoder()
    le.fit(pd.concat([train_data[c],test_data[c]],axis=0))
    name = c+'_encoded'
    train_data[name]= le.transform(train_data.loc[:,c])
    test_data[name]= le.transform(train_data.loc[:,c])
    encoder[c] = le

X = train_data.drop(columns=catagory)
Y = test_data.drop(columns=catagory)
X.drop(columns='msno',inplace=True)
Y.drop(columns='msno',inplace=True)
X = X.iloc[:,1:].values
Y = Y.iloc[:,0].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)
lloss = accuracy_score(y_test, np.zeros_like(y_test))
print("Null LogLoss:  {:.4f}".format(lloss))
print("Baseline Model Training")
start = time.time()
xgb = XGBClassifier(max_depth=10,
                    n_estimator=300,
                    objective='binary:logistic',
                    nthreads=-1)

xgb.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='logloss',
        verbose=1)

y_pred_proba = xgb.predict_proba(X_test)
logloss = log_loss(y_test,y_pred_proba)
y_pred = xgb.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
elapsed = time.time() - start
print("Baseline Model")
print("Accuracy: {:.2f}%".format(accuracy * 100.0))
print("LogLoss:  {:.4f}".format(logloss))
print("Total Time {:.2f}".format(elapsed ))
pickle.dump(xgb, open("xgb_base.pickle.dat", "wb"))

print ("Random Rearch")


import scipy.stats as st
n_iter_search = 60
one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)
params = {  
    "n_estimators": st.randint(100, 1000),
    "max_depth": st.randint(5, 20),
    "learning_rate": st.uniform(0.01, 0.2),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

xgb_rscv = XGBClassifier(nthreads=-1)
random_search = RandomizedSearchCV(xgb_rscv,
                                   param_distributions=params,
                                   cv=3,
                                   scoring='neg_log_loss',
                                   n_iter=n_iter_search,
                                   random_state=0,
                                   n_jobs=-1,
                                   verbose=1)

random_search.fit(X_train, y_train)
pickle.dump(random_search, open("search_result.pickle","wb"))
pickle.dump(random_search.best_model_, open("xgb_rs_best.pickle.dat", "wb"))
best_model = random_search.best_model_


y_pred_proba = best_model.predict_proba(X_test)
logloss = log_loss(y_test,y_pred_proba)
y_pred = best_model.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
elapsed = time.time() - start
print("Baseline Model")
print("Accuracy: {:.2f}%".format(accuracy * 100.0))
print("LogLoss:  {:.4f}".format(logloss))
print("Total Time {:.2f}".format(elapsed ))