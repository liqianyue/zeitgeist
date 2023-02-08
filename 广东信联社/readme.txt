*************************************作品说明*************************************
一、所包含的程序及输出文件

1、特征工程_1.ipynb："train7.9_screening.csv","test7.9_screening.csv","y_train.csv"[数据集1]

2、特征工程_2.ipynb："train8.3.csv","test8.3.csv"[数据集2]

3、lightGBM_1.ipynb："prob_lgb_1.csv","prob_lgb+focal_loss.csv","prob_lgb_2.csv"[预测概率A,B,C]

4、Catboost.ipynb："prob_catboost.csv"[预测概率D]

5、CNN.ipynb："cnn_test_label.csv"[预测概率E]

6、lightGBM_2.ipynb："prob_lgb_4.csv"[预测概率F]

7、XGB_classifier.ipynb："xgb_pro_test.csv"[预测概率G]

8、stack.ipynb："final_test_label.csv"

二、程序执行流程

1、"特征工程_1.ipynb":对原始数据集进行工程处理，输出训练集1，测试集1和训练集标签

2、"特征工程_2.ipynb":对原始数据集进行工程处理，输出训练集2，测试集2

3、"lightGBM_1.ipynb":输入"特征工程_1"处理得到的数据集，用lightGBM算法对模型进行训练，以概率形式预测测试集，输出预测概率；其中使用了三种不同的方法进行训练（不同的模型参数以及新定义的focal_loss目标函数），因此总共输出三个预测概率（A,B,C）

4、"Catboost.ipynb":输入"特征工程_1"处理得到的数据集,用Catboost算法对模型进行训练，以概率形式预测测试集，输出预测概率D

5、"CNN.ipynb":输入"特征工程_1"处理得到的数据集,用CNN算法对模型进行训练，以概率形式预测测试集，输出预测概率E

6、"lightGBM_2.ipynb":输入"特征工程_2"处理得到的数据集，用lightGBM算法对模型进行训练，以概率形式预测测试集，输出预测概率F

7、"XGB_classifier.ipynb":输入"特征工程_2"处理得到的数据集，用xgboost算法对模型进行训练，以概率形式预测测试集，输出预测概率G

8、"stack.ipynb":对输出的七个预测概率A，B，C，D，E，F，G按照阈值输出预测标签，再通过投票法得到最终的预测标签


