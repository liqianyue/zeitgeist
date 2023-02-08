import pandas as pd
import os
import csv
import numpy as np
data_name_list = ['h_b', 'h_k', 'h_l', 'm_b', 'm_h', 'm_k', 'm_l', 'm_t', 'r_b', 'r_k', 'r_l']

def DiNUCindex_RNA(sequence):
    DI_RNA = pd.read_csv("./datasets/DI_RNA.csv", index_col=0)
    pro_index = DI_RNA.index
    sequnece_res = []
    for i in range(len(sequence)-1):
        sequnece_loc = []
        for j in pro_index:
            sequnece_loc.append('%.3f' % DI_RNA.loc[j, sequence[i:i+2]])
        sequnece_res.extend(sequnece_loc)
    return sequnece_res


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers


def N6_math(data_name, data_cla,k_mer):
    if data_cla == 'train':
        data_path = './datasets/benchmark/'+data_name+'_all.fa'
    else:
        data_path = './datasets/independent/'+data_name+'_Test.fa'
    data = pd.read_csv(data_path, header=None)

    data_list = []
    for i in range(0, data.shape[0], 2):
        data_loc = []
        sequence_pro = data.loc[i+1, 0]
        data_loc.append(seq2kmer(sequence_pro.replace("U", "T"), k_mer))
        if '+' in data.loc[i, 0]:
            data_loc.append(1)
        else:
            data_loc.append(0)
        data_list.append(data_loc)

    if not os.path.isdir('./datasets/' + str(k_mer)):
        os.mkdir('./datasets/' + str(k_mer))
    output_path = './datasets/' + str(k_mer) + '/' + data_name
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    output_path = output_path+'/'+data_cla+'.tsv'
    with open(r'{}'.format(output_path), 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerow(['sequence', 'label'])
        tsv_w.writerows(np.array(data_list).tolist())

    pass


def sequence_mer_get(data_frame):
    acid_list = ["A", "C", "G", "U"]
    vocab_dict = {}
    loc = 0
    for i in acid_list:
        for j in acid_list:
            for k in acid_list:
                vocab_loc = i + j + k
                vocab_dict[vocab_loc] = loc
                loc += 1
    sequence_list = []
    for i in range(data_frame.shape[0]):
        sequence_loc = []
        for j in range(len(data_frame.loc[i, "sequence"]) - 2):
            sequence_loc.append(vocab_dict[data_frame.loc[i, "sequence"][j:j + 3]])
        sequence_list.append(sequence_loc)
    return sequence_list


def sequence_mer_get_list(data_frame):
    acid_list = ["A", "C", "G", "U"]
    vocab_dict = {}
    loc = 0
    for i in acid_list:
        for j in acid_list:
            for k in acid_list:
                vocab_loc = i + j + k
                vocab_dict[vocab_loc] = loc
                loc += 1
    sequence_list = []
    for j in range(len(data_frame) - 2):
        sequence_list.append(vocab_dict[data_frame[j:j + 3]])
    return sequence_list


def BERT_extracting(i):
    data_cla_list = ['train', 'dev']
    k_mer_list = [3, 4, 5, 6]
    for j in data_cla_list:
        for k in k_mer_list:
            N6_math(i, j, k)
            print("{}_{}_{}".format(i, j, k))


def sequence_extracting(data_name):

    if not os.path.isdir('./datasets/{}'.format(data_name)):
        os.mkdir('./datasets/{}'.format(data_name))

    data_train = pd.read_csv("./datasets/benchmark/{}_all.fa".format(data_name), header=None)
    data_train_sequence = []
    for i in range(0, data_train.shape[0], 2):
        data_train_sequence.append(sequence_mer_get_list(data_train.loc[i+1, 0]))
    data_train_sequence = pd.DataFrame(data_train_sequence)
    data_train_sequence.to_csv("./datasets/{}/train_sequence.csv".format(data_name), header=None, index=0)

    data_test = pd.read_csv("./datasets/independent/{}_Test.fa".format(data_name), header=None)
    data_test_sequence = []
    for i in range(0, data_test.shape[0], 2):
        data_test_sequence.append(sequence_mer_get_list(data_test.loc[i+1, 0]))
    data_test_sequence = pd.DataFrame(data_test_sequence)
    data_test_sequence.to_csv("./datasets/{}/test_sequence.csv".format(data_name), header=None, index=0)


def Di_extracting(i):
    train_path = "./datasets/benchmark/"
    test_path = "./datasets/independent/"
    print(i)
    output_path = "./datasets/" + i
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    train_path_loc = train_path + i + "_all.fa"
    train_data = pd.read_csv(train_path_loc, header=None)
    train_label_list = []
    data_train_DI = []
    for j in range(0, train_data.shape[0], 2):
        if "+" in train_data.loc[j, 0]:
            train_label_list.append(1)
        else:
            train_label_list.append(0)
        data_train_DI.append(DiNUCindex_RNA(train_data.loc[j + 1, 0]))
    data_train_DI = pd.DataFrame(data_train_DI)
    train_label_list = pd.DataFrame(train_label_list)
    data_train_DI.to_csv(output_path + "/train_DI.csv", header=None, index=None)
    train_label_list.to_csv(output_path + "/train_label.csv", header=None, index=None)

    test_path_loc = test_path + i + "_Test.fa"
    test_data = pd.read_csv(test_path_loc, header=None)
    test_label_list = []
    data_test_DI = []
    for j in range(0, test_data.shape[0], 2):
        if "+" in test_data.loc[j, 0]:
            test_label_list.append(1)
        else:
            test_label_list.append(0)
        data_test_DI.append(DiNUCindex_RNA(test_data.loc[j + 1, 0]))
    data_test_DI = pd.DataFrame(data_test_DI)
    test_label_list = pd.DataFrame(test_label_list)
    data_test_DI.to_csv(output_path + "/test_DI.csv", header=None, index=None)
    test_label_list.to_csv(output_path + "/test_label.csv", header=None, index=None)

