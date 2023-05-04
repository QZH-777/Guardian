import MFCC #これは2で作ったもの
from sklearn.metrics import confusion_matrix
#from sklearn.svm import LinearSVC
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.metrics import classification_report, accuracy_score


if __name__ == '__main__':
    tuned_parameters = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [2, 3, 4], 'gamma': [0.001, 0.0001]},
        {'C': [1, 10, 100, 1000], 'kernel': ['sigmoid'], 'gamma': [0.001, 0.0001]}
        ]
    name_list = ["en_a", "ja_a", "no_voice"]
    train_x, train_y = MFCC.read_ceps(name_list, False)
    test_x, test_y = MFCC.read_ceps(name_list)
    score = 'f1'
    clf = GridSearchCV(
        SVC(), # 識別器
        tuned_parameters, # 最適化したいパラメータセット
        cv=5, # 交差検定の回数
        scoring='%s_weighted' % score ) # モデルの評価関数の指定
    res = clf.fit(train_x, train_y)

    print("score: ", clf.grid_scores_)
    print("best_params:", res.best_params_)
    print("best_estimator:", res.best_estimator_)
