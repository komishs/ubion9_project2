import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# 교차검증 라이브러리
from sklearn.model_selection import StratifiedKFold

# 머신러닝 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 성능평가 라이브러리
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

names = ['lr', 'dt', 'svm', 'rf', 'xgb', 'lgbm', 'cat']

# 성능평가
# output : dictionary
def eval(y_test, y_pred, y_pred_proba, threshold = 0.5, print=False):
    

    cf_matrix = confusion_matrix(y_test, y_pred)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC AUC의 해석
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    ## 0.5: 무작위 예측과 같음. 모델이 아무런 정보를 제공하지 않고 무작위로 예측한 것과 동일합니다.
    ## 0.5 ~ 0.7: 모델의 예측이 약간 낫다고 볼 수 있지만, 그다지 유용하지 않습니다.
    ## 0.7 ~ 0.8: 어느 정도 유용하다고 할 수 있습니다. 모델이 어느 정도 분류를 수행하고 있는 것으로 볼 수 있습니다.
    ## 0.8 ~ 0.9: 상당히 좋은 성능을 나타냅니다. 모델이 잘 분류하고 있으며 실전에 적용할 수 있을 만큼 신뢰할 수 있습니다.
    ## 0.9 이상: 우수한 성능을 나타냅니다. 모델이 매우 잘 분류하고 있으며, 실전에서 사용하기에 아주 신뢰할 수 있습니다.
    
    if print:
        print('Confusion matrix :')
        print(cf_matrix)

        print("Accuracy : %.3f" % acc)
        print("Precision : %.3f" % prec)
        print("Recall : %.3f" % rec)
        print("F1 : %.3f" % f1)

        print('ROC AUC : %.3f' % roc_auc)

    return {
        'cf_matrix' : cf_matrix,
        'accuracy' : acc,
        'precision' : prec,
        'recall' : rec,
        'f1' : f1,
        'roc_auc' : roc_auc
    }

# 교차검증
# output : pd.Dataframe
def skfold(model, X, y, cv=5):
    results = []
    n_iter = 0

    skf = StratifiedKFold(n_splits=cv)

    for train_idx, test_idx in skf.split(X, y):
        n_iter += 1

        X_train = X.iloc[train_idx]
        X_val = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[test_idx]

        # 학습 및 예측
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # 반복시마다 정확도 측정
        results.append(eval(y_val, y_pred, y_pred_proba))

    results = pd.DataFrame(results)
    # for col in results.drop(columns='cf_matrix').columns:
    #     print(f'{col} : {np.mean(results[col], 4)}')

    return results



# 모델 생성
# output : model
def get_model_base(model_name, random_state=42):
    if model_name == 'lr':
        model = LogisticRegression(random_state=random_state)
    elif model_name == 'dt':
        model = DecisionTreeClassifier(random_state=random_state)
    elif model_name == 'svm':
        model = SVC(random_state=random_state, probability=True)
    elif model_name == 'rf':
        model = RandomForestClassifier(random_state=random_state)
    elif model_name == 'xgb':
        model = XGBClassifier(random_state=random_state)
    elif model_name == 'lgbm':
        model = LGBMClassifier(random_state=random_state)
    elif model_name == 'cat':
        model = CatBoostClassifier(random_state=random_state)
    return model

# 모델 학습
# output : dictionary
def train(
        X_train, y_train, X_test, y_test, 
        model_name='lr', model=None, cv=True, cv_splits=5
):
    if model is None:
        model = get_model_base(model_name)
    model.fit(X_train, y_train)

    # 예측
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    results = {
        'name' : model_name,
        'model' : model, 
        'results' : eval(y_test, y_pred, y_pred_proba)
    }

    if cv:
        results['cv'] = skfold(model, X_train, y_train, cv=cv_splits)

    return results