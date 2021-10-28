1_train_test_split.ipynb	按照4：1的比例对数据集进行分层抽样，得到训练集和独立测试集

2_gene_importance.ipynb	使用树模型中的特征排序方法，对全部基因进行重要性排序并输出重要性程度

3_importance_sort_accuracy.ipynb	使用五种常用的模型，在分别取不同数量基因时，进行训练，然后使用五折交叉验证的方式进行评估模型效果，最终确定基因选择的数量

4_gene_sel_plot.ipynb	对第三步得到的模型效果，进行可视化展示

5_GEO_xgboost.ipynb和6_TCGA_xgboost.ipynb	确定基因选择结果后，使用五折交叉验证对比五种模型的预测效果，并对xgboost模型的超参数进行寻优；并保存该步骤训练所有模型的参数

7_test_evaluation.ipynb	对之前训练的模型，进行独立测试集评估，并输出多种评价指标值，多方面对比模型效果，最后输出xgboost模型对每个癌种的预测效果。
			