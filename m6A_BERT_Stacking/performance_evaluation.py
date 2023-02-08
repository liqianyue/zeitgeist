import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, confusion_matrix, matthews_corrcoef, roc_auc_score


def eval_funs(pre_pro_pri, pre_ture):
    pre_pro = []
    pre_label = []
    for i in pre_pro_pri:
        pre_pro.append(i[0, 1])
        if i[0, 1] >= 0.5:
            pre_label.append(1)
        else:
            pre_label.append(0)

    acc = accuracy_score(pre_ture, pre_label)
    me = confusion_matrix(pre_ture, pre_label)
    mcc = matthews_corrcoef(pre_ture, pre_label)
    TN, FN, FP, TP = me[0, 0], me[1, 0], me[0, 1], me[1, 1]
    a = roc_auc_score(pre_ture, pre_pro)

    # Sensitivity
    Sn = TP / (TP + FN)
    # Specificity
    Sp = TN / (TN + FP)

    print("Accuracy {:.3f}".format(acc))
    print("Sensitivity {:.3f}".format(Sn))
    print("Specificity {:.3f}".format(Sp))
    print("matthews_corrcoef {:.3f}".format(mcc))
    print("AUROC: {:.3f}".format(a))
