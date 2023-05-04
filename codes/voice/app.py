import MFCC #これは2で作ったもの
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import resample
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

def main():
    name_list = ["en_a", "ja_a", "no_voice"]
    train_x, train_y = MFCC.read_ceps(name_list, False)
    test_x, test_y = MFCC.read_ceps(name_list, True)
    svc = SVC(kernel="linear",
        C=100.0,
        degree=3,
        gamma='auto',
        cache_size=200,
        class_weight=None,
        coef0=0.0,
        decision_function_shape=None,
        max_iter=-1,
        probability=False,
        random_state=None,
        shrinking=True,
        tol=0.001,
        verbose=False)
    svc.fit(train_x, train_y)
    #print(train_x, train_y)

    prediction_y = svc.predict(test_x)
    print(prediction_y)
    print(test_y)
    cm = confusion_matrix(test_y, prediction_y)

    acc_parcent = accuracy_score(test_y, prediction_y)
    print(acc_parcent)
    print(cm)

if __name__ == '__main__':
    main()
