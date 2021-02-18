import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline



# PREPARE TRAINING DATA AND SAVE ENCODER
df_train = pd.read_csv('adult.data', sep=', ')

df_train.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50K']


df_train['>50K'] = df_train['>50K'].map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})

df_train['workclass'][df_train['workclass'] == '?'] = 'Not Telling'
df_train['occupation'][df_train['occupation'] == '?'] = 'Not Telling'

X_train_cat = df_train[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']].values
X_train_num = df_train[['age', 'hours-per-week']].values


workclass = np.unique(X_train_cat[:, 0])
education = np.unique(X_train_cat[:, 1])
marital_status = np.unique(X_train_cat[:, 2])
occupation = np.unique(X_train_cat[:, 3])
relationship = np.unique(X_train_cat[:, 4])
race = np.unique(X_train_cat[:, 5])
sex = np.unique(X_train_cat[:, 6])
native_country = np.unique(X_train_cat[:, 7])

ohe = OneHotEncoder(categories=[workclass, education, marital_status, occupation, relationship, race, sex, native_country])
X_train_cat_ohe = ohe.fit_transform(X_train_cat).toarray()
with open("encoder", "wb") as f:
    pickle.dump(ohe, f)

X_train = np.concatenate([X_train_cat_ohe, X_train_num], axis=1)
y_train = df_train['>50K'].values


tuning = False
if tuning:
    rf = RandomForestClassifier(random_state=42)

    param_grid_rand = {'n_estimators': randint(20, 200),
                       'max_depth': [k for k in range(8, 20)],
                       'min_samples_leaf': [0, 5, 10, 30]
                       }

    random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid_rand,
                                       scoring='roc_auc', n_iter=20, cv=5, verbose=3,
                                       random_state=42)

    random_search.fit(X_train, y_train)

    print("Best roc_auc_score {:.6f} params {}".format(-random_search.best_score_, random_search.best_params_))
    # result : Best roc_auc_score -0.889260 params {'max_depth': 19, 'min_samples_leaf': 5, 'n_estimators': 28}

    exit(0)

model = RandomForestClassifier(max_depth=19, min_samples_leaf=5, n_estimators=28, random_state=42)
model.fit(X_train, y_train)


# LOAD AND TRANSFORM TEST DATA AND COMPUTE TEST ROC AUC
df_test = pd.read_csv('adult.test', sep=', ')

df_test.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                    'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', '>50K']

df_test['>50K'] = df_test['>50K'].map({'<=50K': 0, '<=50K.': 0, '>50K': 1, '>50K.': 1})
df_test['workclass'][df_test['workclass'] == '?'] = 'Not Telling'
df_test['occupation'][df_test['occupation'] == '?'] = 'Not Telling'

X_test_cat = df_test[['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']].values
X_test_num = df_test[['age', 'hours-per-week']].values

with open('encoder', 'rb') as file:
    encoder = pickle.load(file)

print(X_test_cat.shape)
X_test_cat_ohe = encoder.fit_transform(X_test_cat).toarray()
X_test = np.concatenate([X_test_cat_ohe, X_test_num], axis=1)
print(X_test.shape)
y_test = df_test['>50K'].values

metrics.plot_roc_curve(model, X_test, y_test)
pickle.dump(model, open('income_predictor.pkl', 'wb'))
plt.show()


