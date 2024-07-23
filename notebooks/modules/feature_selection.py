import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import scipy.stats as stats
from scipy.stats import bartlett

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm

from . import modeling as ml



# 등분산성 검정
def bartlett_test(df, col_y, col, p_value = 0.05, H = 1):
    # H가 1인 경우 이분산성 / H가 0인 경우 등분산
    list= []

    for i in col:
        T, p_val = bartlett(df[df[col_y]==1][i], df[df[col_y]==0][i]) 
        list.append([i, p_val])

    list = pd.DataFrame(list, columns = ['변수', 'p_value'])
    
    if H == 1:
        a = list[(list['p_value'] < p_value)][['변수', 'p_value']].sort_values('p_value')
        return a
    else:
        a = list[(list['p_value'] >= p_value)][['변수', 'p_value']].sort_values('p_value')
        return a
    
# T-Test 검정
def t_test(df, col_y, col, col_h0, col_h1, p_value = 0.05):
    list= []
    for i in col:
        
        data1 = df[df[col_y]==1][i] # label=1인 집단의 i 피처 데이터
        data0 = df[df[col_y]==0][i] # label=0인 집단의 i 피처 데이터

        if (col_h0['변수']==i).any():
            # 등분산성 : wald t-test
            t_stat, p_val = stats.ttest_ind(
                data1, data0, equal_var=True
            ) 
            list.append([i, 'homo', p_val])
        elif (col_h1['변수']==i).any():
            # 이분산성 : welch’s t-test
            t_stat, p_val = stats.ttest_ind(
                data1, data0, equal_var=False
            )
            list.append([i, 'hetero', p_val])

    list = pd.DataFrame(list, columns = ['변수', '분산', 'p_value'])

    # pvalue < 0.05인 피처 데이터만 return
    a = list[(list['p_value'] < p_value)][['변수', '분산', 'p_value']].sort_values('p_value')
    return a

# 피처 셀렉션 : Filter (t-test)
def selection_ttest(train, col_y = 'label', cols_feature = None, pvalue=0.05):
    if cols_feature is None:
        cols_feature = train.columns.difference([col_y])

    """ (1) 등분산성 확인 """

    ## 이분산성 변수
    x_hetero = bartlett_test(train, col_y, cols_feature, H = 1)

    ## 등분산성 변수
    x_homo = bartlett_test(train, col_y, cols_feature, H = 0)
    
    """ (2) T-test 적용 """

    # 2) t_test 결과 p_value < 0.05보다 작은 유의한 변수 가져오기
    x_ttest = t_test(train, col_y, cols_feature, x_homo, x_hetero, p_value=pvalue)

    print("후보 피처 수 : ", len(cols_feature)-1)
    print("유의한 피쳐 수 :", len(x_ttest))

    return x_ttest.sort_values(by="p_value", ascending=True)

# 피처 셀렉션 : Wrapper (backward)
def selection_backward(
        X_train, y_train, X_test, y_test,
        cols_feature=None,
        base_model_name='lr'
):
    if cols_feature is None:
        cols_feature = X_train.columns
    
    # Backward feature selection 수행
    selected_features = cols_feature
    best_score = 0
    while len(selected_features) > 0:
        worst_feature = None
        best_score_local = 0

        for feature in selected_features:
            features = selected_features.copy()
            features.remove(feature)

            X_train_selected = X_train[features]
            X_test_selected = X_test[features]

            # model = LogisticRegression()
            model = ml.get_model_base(model_name=base_model_name)

            model.fit(X_train_selected, y_train)
            score = model.score(X_test_selected, y_test)

            if score > best_score_local:
                best_score_local = score
                worst_feature = feature

        if best_score_local > best_score:
            selected_features.remove(worst_feature)
            best_score = best_score_local
            print(f"Removed feature: {worst_feature}, Accuracy: {best_score:.4f}")
        else:
            break

    print("\nFinal selected features:")
    Backward = selected_features
    print(f'선택된 피처 수: {len(Backward)}')
    return Backward

# 피처 셀렉션 : Embedded (logit)
def selection_logit(X_train, y_train):
    model = LogisticRegression(random_state=42)

    logit = SelectFromModel(model)
    logit.fit(X_train, y_train)

    logit_support = logit.get_support()
    lr_feature = X_train.loc[:,logit_support].columns.tolist()

    print('Logit으로 선택된 피처 수: ', len(lr_feature))
    return lr_feature