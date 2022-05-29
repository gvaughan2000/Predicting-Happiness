import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


def make_models(X_train, y_train):
    # Logistic regression
    LR = LogisticRegression(solver='lbfgs', multi_class='multinomial')

    # SVM
    SVM_ovo = svm.SVC(decision_function_shape="ovo")
    SVM_linear = svm.SVC(kernel='linear')

    # Random Forest
    RF = RandomForestClassifier(n_estimators=1000, max_depth=10)

    # Multilayer Perceptron
    NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10))

    ## Gradient Boost
    GB = GradientBoostingClassifier(n_estimators=1000, max_depth=10)

    # XGB
    XGB = XGBClassifier(objective='binary:logistic')

    ## Light Gradient Boost
    LGB = LGBMClassifier()

    ## CatBoost
    CatBoost = CatBoostClassifier(verbose=0, n_estimators=100)

    ## Decision Tree
    DT = tree.DecisionTreeClassifier(max_depth=10)

    # K nearest neighbours
    KNN = KNeighborsClassifier(n_neighbors=400)

    ## Gaussian Native Bayes
    gnb = GaussianNB().fit(X_train, y_train)

    all_models = {'Logistic Regression': LR,
                  'SVM ovo': SVM_ovo,
                  'SVM linear': SVM_linear,
                  'Random Forest': RF,
                  'Neural Network': NN,
                  'Decision Tree': DT,
                  'Gradient Boost': GB,
                  'ExtremeGradientBoost': XGB,
                  'K nearest neighbours': KNN,
                  'Light Gradient Boost': LGB,
                  'Cat Boost': CatBoost,
                  'Gaussian Native Bayes': gnb}

    return all_models


def compare_models(models_dict, X_train, y_train, X_test, y_test):
    compare_df = pd.DataFrame(columns=['Name', 'Score', 'RMSE', 'Accuracy', 'Precision', 'Recall'])

    for name, model in models_dict.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred).ravel()

        model_dict = {
            'Name': name,
            'Score': round(model.score(X_test, y_test), 3),
            'RMSE': round(metrics.mean_squared_error(y_test, y_pred, squared=False), 3),
            "Accuracy": round(metrics.accuracy_score(y_test, y_pred), 3),
            "Precision": round(metrics.precision_score(y_test, y_pred), 3),
            "Recall": round(metrics.recall_score(y_test, y_pred), 3),
            "True_N": round(tn),
            "True_P": round(tp),
            "False_N": round(fn),
            "False_P": round(fp),
        }

        compare_df = compare_df.append(model_dict, ignore_index=True)

    compare_df = compare_df.sort_values('Score', ascending=False).reset_index(drop=True)
    return compare_df