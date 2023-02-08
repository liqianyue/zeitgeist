import os
import shutil

path_data = './datasets/'


def run_main(class_name, DNA_mer, Batch_size, Epochs):

    data_dir = "{}/{}".format(path_data + DNA_mer, class_name)
    output_dir = "./BERT_m6a/examples/fine_models/{}/{}".format(DNA_mer, class_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_run = "python run_finetune.py " \
                "--model_type dna --tokenizer_name=dna{} --model_name_or_path ./BERT_m6a/examples/pre_models/output{} " \
                "--task_name dnaprom --do_train --do_eval " \
                "--data_dir {} " \
                "--max_seq_length 100 " \
                "--per_gpu_eval_batch_size={} --per_gpu_train_batch_size={}  " \
                "--learning_rate 1e-5 " \
                "--num_train_epochs {} " \
                "--output_dir {} " \
                "--evaluate_during_training --logging_steps 1070 --save_steps 8000 " \
                "--warmup_percent 0.1 --hidden_dropout_prob 0.1 --overwrite_output " \
                "--weight_decay 0.0001 --n_process 1".format(DNA_mer, DNA_mer, data_dir, Batch_size, Batch_size, Epochs, output_dir)

    pre_val_path = "{}/{}_val".format(data_dir, class_name)
    pre_test_path = "{}/{}_test".format(data_dir, class_name)
    predict_val_path = "./BERT_m6a/examples/result/{}/{}_val".format(DNA_mer, class_name)
    predict_test_path = "./BERT_m6a/examples/result/{}/{}_test".format(DNA_mer, class_name)

    dev_run_val = "python run_finetune.py " \
              "--model_type dna --tokenizer_name=dna{} " \
              "--model_name_or_path {} " \
              "--task_name dnaprom --do_predict " \
              "--data_dir {}  --max_seq_length 100 " \
              "--per_gpu_pred_batch_size={}  " \
              "--output_dir {} " \
              "--predict_dir {} " \
              "--n_process 48".format(DNA_mer, output_dir, pre_val_path, Batch_size, output_dir, predict_val_path)

    dev_run_test = "python run_finetune.py " \
              "--model_type dna --tokenizer_name=dna{} " \
              "--model_name_or_path {} " \
              "--task_name dnaprom --do_predict " \
              "--data_dir {}  --max_seq_length 100 " \
              "--per_gpu_pred_batch_size={}  " \
              "--output_dir {} " \
              "--predict_dir {} " \
              "--n_process 48".format(DNA_mer, output_dir, pre_test_path, Batch_size, output_dir, predict_test_path)

    # train_run / dev_run_val / dev_run_test
    os.system(train_run)
    os.system(dev_run_val)
    os.system(dev_run_test)

    shutil.move("./BERT_m6a/examples/result/{}/{}_val/pred_results.npy".format(DNA_mer, class_name),
                      "{}/{}/pred_results_train.npy".format(path_data + DNA_mer, class_name))
    shutil.move("./BERT_m6a/examples/result/{}/{}_test/pred_results.npy".format(DNA_mer, class_name),
                      "{}/{}/pred_results_test.npy".format(path_data + DNA_mer, class_name))
