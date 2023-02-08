import shutil
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import os

path_data = './datasets/'


def data_cv_class(class_name):

    k_mer_list = ["3", "4", "5", "6"]
    for k_mer in k_mer_list:
        os.mkdir(path_data + "{}/{}/{}_val".format(k_mer, class_name, class_name))
        os.mkdir(path_data + "{}/{}/{}_test".format(k_mer, class_name, class_name))
        os.rename(path_data + "{}/{}/train.tsv".format(k_mer, class_name),
                  path_data + "{}/{}/train_pri.tsv".format(k_mer, class_name))
        shutil.move(path_data + "{}/{}/dev.tsv".format(k_mer, class_name),
                    path_data + "{}/{}/{}_test/dev.tsv".format(k_mer, class_name, class_name))

    for k_mer in k_mer_list:
        data_train = pd.read_csv(path_data + "{}/{}/train_pri.tsv".format(k_mer, class_name), sep='\t')
        data_train_label = data_train.loc[:, 'label'].values
        data_train = data_train.loc[:, 'sequence'].values
        X_train, X_test, y_train, y_test = train_test_split(data_train, data_train_label, test_size=0.2, random_state=20)

        data_train_list = []
        for i in range(X_train.shape[0]):
            data_train_list.append([X_train[i], y_train[i]])

        data_val_list = []
        for i in range(X_test.shape[0]):
            data_val_list.append([X_test[i], y_test[i]])

        with open(r'{}/'.format(path_data + k_mer) + "/" + class_name + '/train.tsv', 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['sequence', 'label'])
            tsv_w.writerows(np.array(data_train_list).tolist())

        with open(r'{}/'.format(path_data + k_mer) + "/" + class_name + '/dev.tsv', 'w', newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['sequence', 'label'])
            tsv_w.writerows(np.array(data_val_list).tolist())
        with open(r'{}/'.format(path_data + k_mer) + "/" + class_name + "/{}_val/".format(class_name) + '/dev.tsv', 'w',
                  newline='') as f:
            tsv_w = csv.writer(f, delimiter='\t')
            tsv_w.writerow(['sequence', 'label'])
            tsv_w.writerows(np.array(data_val_list).tolist())

