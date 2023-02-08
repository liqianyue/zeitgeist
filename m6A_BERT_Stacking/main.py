from data_extract import BERT_extracting, sequence_extracting, Di_extracting
from BERT_m6a.examples.data_cv import data_cv_class
from BERT_m6a.examples.run_text import run_main
from Resnet.classifier_mian import resnet_main, BiLSTM_main, Linear_full_main, performance


if __name__ == '__main__':
    data_name_list = ['h_b', 'h_k', 'h_l', 'm_b', 'm_h', 'm_k', 'm_l', 'm_t', 'r_b', 'r_k', 'r_l']

    # # 1.feature extract part
    # for data_name in data_name_list:
    #     BERT_extracting(data_name)
        # sequence_extracting(data_name)
        # Di_extracting(data_name)

    # # 2.DNABERT training part
    # data_name = 'h_b'   # training dataset
    # k_mer = "3"         # DNABERT k_mer
    # batch_size = 12
    # epochs = 2
    # # data_cv_class(data_name)           # split training set and valication set
    # run_main(data_name, k_mer, batch_size, epochs)  # running

    # # 3.Resnet trainning part
    data_name = 'm_b'
    batch_size = 16
    epochs = 40
    learning_rate = 1e-4
    best_acc = 0.0
    resnet_main(data_name, epochs, batch_size, learning_rate, best_acc)
    #
    # # # 4.BiLSTM trainning part
    # data_name = 'm_b'
    # batch_size = 16
    # epochs = 40
    # learning_rate = 1e-4
    # best_acc = 0.0
    # BiLSTM_main(data_name, epochs, batch_size, learning_rate, best_acc)
    #
    # # # 5.Full linear layer trainning part
    # data_name = 'm_b'
    # batch_size = 16
    # epochs = 40
    # learning_rate = 1e-4
    # best_acc = 0.0
    # Linear_full_main(data_name, epochs, batch_size, learning_rate, best_acc)
    #
    # # # 6.performance
    # performance(data_name, "linear_full_model")
    pass




