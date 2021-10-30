1_train_test_split.ipynb	Stratified sampling of the data set in the ratio of 4:1 to obtain the training set and the independent test set

2_gene_importance.ipynb	Use the feature ranking method in the tree model to rank the importance of all genes and output the importance level

3_importance_sort_accuracy.ipynb	Five commonly used models are trained using different numbers of genes when taken separately, and then the model effects are evaluated using a five-fold cross-validation to finally determine the number of genes to select

4_gene_sel_plot.ipynb	Visualization of the model effect obtained in step 3

5_GEO_xgboost.ipynb和6_TCGA_xgboost.ipynb	After determining the gene selection results, compare the prediction effects of the five models using five-fold cross-validation, and perform optimization search for the hyperparameters of the xgboost model; and save the parameters of all models trained in this step

7_test_evaluation.ipynb	The previously trained models are evaluated in an independent test set, and multiple evaluation index values are output to compare the model effects in many aspects, and finally the prediction effect of xgboost model for each cancer species is output.
			
