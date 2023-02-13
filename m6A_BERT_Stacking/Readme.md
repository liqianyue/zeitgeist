M6A-BERT-Stacking: A tissue-specific predictor for identifying RNA N6-methyladenosine sites based on BERT and stacking strategy
===
## Requirements
* Python == 3.8.3
* conda == 4.12.0
* torch == 1.12.0+cu113
* scikit-learn == 0.24.0
* pandas == 1.0.5
* numpy == 1.19.5
* matplotlib == 3.3.4

Dependencies can be installed using the following command:
```
conda create -n M6A-BERT-Stacking python=3.8.3
conda activate M6A-BERT-Stacking

pip install -r requirements.txt
```

* CUDA == 11.6 (This is just a suggestion to make sure your program works properly)
* how to install CUDA and cuDNN:
```
conda install cudatoolkit=11.6
conda install cudnn=7.6.3
```

## How to use
Train the model
```
cd ./m6A-BERT_Stacking
python main.py --train 1
```
test and show performance
```
cd ./m6A-BERT_Stacking
python main.py --test 1
```
For more parameter information, please refer to main.py.

## The reseach of DNABERT
[DNABERT Github](https://github.com/jerryji1993/DNABERT)

## Results
performance  
  dataset name  | Acc | AUC  
  ------------- | -------------  | -------------
 H_b  |0.827 |  0.806
 H_k  |0.888 |0.815
 H_l  |0.89 |0.792
 M_b  |0.876|0.757
 M_h  |0.835|0.747
 M_k  |0.819|0.898
 M_l  |0.736 |0.816
 M_t  |0.78|0.867
 R_b  |0.783 |0.866
 R_l  |0.838 |0.914
 R_t  |0.82 |0.903

## Citation
If you find this repository useful in your research, please consider citing the Github:
https://github.com/liqianyue/zeitgeist/edit/master/m6A_BERT_Stacking

Papers involving m6A-BERT_Stacking have been submitted to an academic journal.

## Contact
If you have any questions, please feel free to contact Shihang Wang (Email: l15178253360@126.com).

Pull requests are highly welcomed!

## Acknowledgments
Thanks to  Shanghai Ocean University for providing computing infrastructure.
Thank you all for your attention to this work.
















