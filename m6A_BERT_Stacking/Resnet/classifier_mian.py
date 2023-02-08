import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import preprocessing
from torch import Tensor
from tqdm import tqdm
import pandas as pd
import numpy as np
from resnet_cbam import resnet50_cbam
import lstm_att_model
from linear_model import linear_net
from sklearn.model_selection import train_test_split
from performance_evaluation import eval_funs

path_ori_data = './datasets/'


def data_get(data_name, val_test):
    path_data = path_ori_data+data_name+"/"
    zscore_score = preprocessing.StandardScaler()

    data_train = pd.read_csv(path_data + "train_DI.csv", header=None)
    data_train = zscore_score.fit_transform(data_train.values)
    data_train = data_train.reshape(data_train.shape[0], 1, -1, 22)
    data_train_label = pd.read_csv(path_data+"train_label.csv", header=None)
    data_train_label = data_train_label.loc[:, 0].values.tolist()

    data_train, data_val, data_train_label, data_val_label = train_test_split(data_train, data_train_label,
                                                                              test_size=0.2, random_state=20)
    data_val_label_list = pd.DataFrame(data_val_label)
    data_val_label_list.to_csv(path_data+"train_pro_label.csv", header=None, index=0)
    print("val label output success!")

    data_dev = pd.read_csv(path_data + "test_DI.csv", header=None)
    data_dev = zscore_score.transform(data_dev.values)
    data_dev = data_dev.reshape(data_dev.shape[0], 1, -1, 22)
    data_dev_label = pd.read_csv(path_data + "test_label.csv", header=None)
    data_dev_label = data_dev_label.loc[:, 0].values.tolist()

    print(data_train.shape)
    print(data_dev.shape)
    print(data_val.shape)
    print(len(data_train_label))
    print(len(data_dev_label))
    print(len(data_val_label))

    data_train_res = []
    for i in range(data_train.shape[0]):
        data_loc = [(torch.from_numpy(data_train[i])).float(), data_train_label[i]]
        data_train_res.append(data_loc)

    data_dev_res = []
    for i in range(data_dev.shape[0]):
        data_loc = [(torch.from_numpy(data_dev[i])).float(), data_dev_label[i]]
        data_dev_res.append(data_loc)

    data_val_res = []
    for i in range(data_val.shape[0]):
        data_loc = [(torch.from_numpy(data_val[i])).float(), data_val_label[i]]
        data_val_res.append(data_loc)

    if val_test == "val":
        return data_train_res, data_val_res
    elif val_test == "test":
        return data_val_res, data_dev_res


def data_get_sequence(data_name, val_test):
    path_data = path_ori_data + data_name + "/"

    data_train = pd.read_csv(path_data+"train_sequence.csv", header=None)
    data_train = data_train.loc[:, :].values
    data_train =data_train.reshape(data_train.shape[0], 39)
    data_train_label = pd.read_csv(path_data + "train_label.csv", header=None)
    data_train_label = data_train_label.loc[:, 0].values.tolist()

    data_train, data_val, data_train_label, data_val_label = train_test_split(data_train, data_train_label,
                                                                              test_size=0.2, random_state=20)

    data_dev = pd.read_csv(path_data+"test_sequence.csv", header=None)
    data_dev = data_dev.loc[:, :].values
    data_dev = data_dev.reshape(data_dev.shape[0], 39)
    data_dev_label = pd.read_csv(path_data + "test_label.csv", header=None)
    data_dev_label = data_dev_label.loc[:, 0].values.tolist()

    print(data_train.shape)
    print(data_dev.shape)
    print(data_val.shape)
    print(len(data_train_label))
    print(len(data_dev_label))
    print(len(data_val_label))

    data_train_res = []
    for i in range(data_train.shape[0]):
        data_loc = [(torch.from_numpy(data_train[i])).int(), data_train_label[i]]
        data_train_res.append(data_loc)

    data_dev_res = []
    for i in range(data_dev.shape[0]):
        data_loc = [(torch.from_numpy(data_dev[i])).int(), data_dev_label[i]]
        data_dev_res.append(data_loc)

    data_val_res = []
    for i in range(data_val.shape[0]):
        data_loc = [(torch.from_numpy(data_val[i])).int(), data_val_label[i]]
        data_val_res.append(data_loc)

    if val_test == "val":
        return data_train_res, data_val_res
    elif val_test == "test":
        return data_val_res, data_dev_res


def data_get_pro(data_name):
    path_data = path_ori_data + data_name + "/"

    data_train_lstm = pd.read_csv(path_data + data_name + "_bilstm_train_pro.csv", header=None)
    data_train_lstm = data_train_lstm.values
    data_test_lstm = pd.read_csv(path_data + data_name + "_bilstm_test_pro.csv", header=None)
    data_test_lstm = data_test_lstm.values

    data_train_resnet = pd.read_csv(path_data + data_name +"_resnet50_cbam_train_pro.csv", header=None)
    data_train_resnet = data_train_resnet.values
    data_test_resnet = pd.read_csv(path_data + data_name + "_resnet50_cbam_test_pro.csv", header=None)
    data_test_resnet = data_test_resnet.values

    data_train_bert = np.load(path_data + "pred_results_train.npy", encoding="latin1")
    data_test_bert = np.load(path_data + "pred_results_test.npy", encoding="latin1")

    data_train_label = pd.read_csv(path_data + "train_pro_label.csv", header=None)
    data_train_label = data_train_label.loc[:, 0].values.tolist()
    data_dev_label = pd.read_csv(path_data + "test_label.csv", header=None)
    data_dev_label = data_dev_label.loc[:, 0].values.tolist()

    if data_train_lstm.shape[0] < data_train_bert.shape[0]:
        short_num = int(data_train_bert.shape[0] - data_train_lstm.shape[0])
        lstm_num = np.zeros((short_num, 2))
        lstm_num += 0.5
        print("lstm shape short numpy : {}".format(short_num))
        data_train_lstm = np.vstack((data_train_lstm, lstm_num))

    if data_test_lstm.shape[0] < data_test_bert.shape[0]:
        short_num = int(data_test_bert.shape[0] - data_test_lstm.shape[0])
        lstm_num = np.zeros((short_num, 2))
        lstm_num += 0.5
        print("lstm test shape short numpy : {}".format(short_num))
        data_test_lstm = np.vstack((data_test_lstm, lstm_num))

    data_train = np.hstack((data_train_bert, data_train_resnet, data_train_lstm))
    print(data_train.shape)
    data_test = np.hstack((data_test_bert, data_test_resnet, data_test_lstm))
    print(data_test.shape)

    data_train_res = []
    for i in range(data_train.shape[0]):
        data_loc = [(torch.from_numpy(data_train[i])).float(), data_train_label[i]]
        data_train_res.append(data_loc)

    data_dev_res = []
    for i in range(data_test.shape[0]):
        data_loc = [(torch.from_numpy(data_test[i])).float(), data_dev_label[i]]
        data_dev_res.append(data_loc)

    return data_train_res, data_dev_res


def trainning(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc, name_net, Save_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = Batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    if name_net == "bilstm":
        train_dataset, test_dataset = data_get_sequence(class_name, "val")
    elif name_net == "resnet50_cbam":
        train_dataset, test_dataset = data_get(class_name, "val")
    elif name_net == "linear_full_model":
        train_dataset, test_dataset = data_get_pro(class_name)
    else:
        print("data get function error , check your model name")
        return

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw, drop_last=True)

    train_num = len(train_dataset)
    val_num = len(test_dataset)
    validate_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw, drop_last=True)

    print("using {} images for training, {} images for validation.".format(train_num,

                                                                           val_num))
    # net choose
    if name_net == "resnet50_cbam":
        net = resnet50_cbam()
    elif name_net == "bilstm":
        LSTM_hidden = 32
        Embed_dim = 16
        linear_num = 6

        batch_size = Batch_size
        seed = 1111
        cuda_able = True
        vocab_size = 64
        dropout = 0.5
        embed_dim = Embed_dim
        hidden_size = LSTM_hidden
        bidirectional = True
        weight_decay = 0.001
        attention_size = 39
        sequence_length = 39
        use_cuda = torch.cuda.is_available() and cuda_able
        torch.manual_seed(seed)
        output_size = 2

        net = lstm_att_model.bilstm_attn(batch_size=batch_size,
                                         output_size=output_size,
                                         hidden_size=hidden_size,
                                         embed_dim=embed_dim,
                                         vocab_size=vocab_size,
                                         bidirectional=bidirectional,
                                         dropout=dropout,
                                         use_cuda=use_cuda,
                                         attention_size=attention_size,
                                         sequence_length=sequence_length)
    elif name_net == "linear_full_model":
        linear_num = 6
        net = linear_net(linear_num, 2)

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=Learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    epochs = Epoch_num
    best_acc = Best_acc
    save_path = '{}.pth'.format(Save_path)
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.4f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        optimizer.step()
        scheduler.step()
        print(epoch, scheduler.get_last_lr())

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.4f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate >= best_acc:
            best_acc = val_accurate
            # torch.save(net.state_dict(), save_path)
            torch.save(net, save_path)
    print('Finished Training')


def predict_pro(data_name, net_name, Batch_size):
    pthfile = r'./Resnet/model_pth/{}_{}.pth'.format(data_name, net_name)
    net = torch.load(pthfile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    drop_last = False
    batch_size = 1

    if net_name == "bilstm":
        train_dataset, test_dataset = data_get_sequence(data_name, "test")
        drop_last = True
        batch_size = Batch_size
    elif net_name == "resnet50_cbam":
        train_dataset, test_dataset = data_get(data_name, "test")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    validate_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last)

    pro_train = []
    for data in train_loader:
        image, label = data
        output = net(image.to(device))
        pro_train.append(output)

    pro_test = []
    for data in validate_loader:
        image, label = data
        output = net(image.to(device))
        pro_test.append(output)

    pro_train = [aa.tolist() for aa in pro_train]
    pro_train = pd.DataFrame(np.array(pro_train).reshape(-1, 2))
    print(pro_train.shape)
    pro_train.to_csv("{}/{}_{}_train_pro.csv".format(path_ori_data + data_name, data_name, net_name), header=None, index=0)
    pro_test = [aa.tolist() for aa in pro_test]
    pro_test = pd.DataFrame(np.array(pro_test).reshape(-1, 2))
    print(pro_test.shape)
    pro_test.to_csv("{}/{}_{}_test_pro.csv".format(path_ori_data + data_name, data_name, net_name), header=None, index=0)


def performance(data_name, class_name):
    softmax = torch.nn.Softmax(dim=1)
    _, test_dataset = data_get_pro(class_name)
    pthfile = r'./Resnet/model_pth/{}_linear_full_model.pth'.format(data_name)
    net = torch.load(pthfile)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    pro_test = []
    pro_test_label = []
    for data in test_loader:
        image, label = data
        output = net(image.to(device))
        pro_test.append(softmax(Tensor.cpu(torch.tensor(output, dtype=torch.float32))).numpy())
        pro_test_label.append(int(label))

    pro_test = np.array(pro_test)
    eval_funs(pro_test, pro_test_label)


def resnet_main(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc):
    name_net = "resnet50_cbam"
    Save_path = "./Resnet/model_pth/{}_{}".format(class_name, name_net)

    trainning(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc, name_net, Save_path)
    with torch.no_grad():
        predict_pro("{}".format(class_name), "resnet50_cbam")


def BiLSTM_main(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc):
    name_net = "bilstm"
    Save_path = "./Resnet/model_pth/{}_{}".format(class_name, name_net)
    trainning(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc, name_net, Save_path)
    with torch.no_grad():
        predict_pro("{}".format(class_name), "resnet50_cbam")


def Linear_full_main(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc):
    name_net = "linear_full_model"
    Save_path = "./Resnet/model_pth/{}_{}".format(class_name, name_net)
    trainning(class_name, Epoch_num, Batch_size, Learning_rate, Best_acc, name_net, Save_path)
