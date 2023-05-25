# DAPA
Code for our ACL 2023 Findings paper: Domain Aligned Prefix Averaging for Domain Generalization in Abstractive Summarization
<hr>

## Environment Settings
First, create the following environments settings:

```
virtualenv --python=python3 env         # for creatig prefix checkpoints required for DAPA
source env/bin/activate
cd OpenPrompt4
python setup.py install
cd ../transformers
pip install -e ./
deactivate

virtualenv --python=python3 env2        # for running DAPA
source env2/bin/activate
cd OpenPrompt8
python setup.py install
cd ../transformers
pip install -e ./
deactivate
```
<hr>

## Prefix Checkpoints
Next, create prefix checkpoints by running the following:

```
source env/bin/activate
mkdir prefix_checkpoints
cd prefix_checkpoints
mkdir scientific
mkdir cnn
mkdir samsum
mkdir reddit
cd ..

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds.py --train_file scientific_data_AIC_train.pickle --dev_file scientific_data_AIC_dev.pickle --model_name t5-small --store prefix_checkpoints/scientific --print_every 100 --eval_every 500 --file embed_store/scientific_data_AIC50_train_embed.pth --num_token 50 --bs 5

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds.py --train_file cnn_train.pickle --dev_file cnn_dev.pickle --model_name t5-small --store prefix_checkpoints/cnn --print_every 1000 --eval_every 28000 --file embed_store/cnn50_train_embed.pth --num_token 50 --bs 10

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds.py --train_file samsum_train.pickle --dev_file samsum_dev.pickle --model_name t5-small --store prefix_checkpoints/samsum --print_every 1000 --eval_every 5000 --file embed_store/samsum50_train_embed.pth --num_token 50 --bs 5

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds.py --train_file reddit_train.pickle --dev_file reddit_dev.pickle --model_name t5-small --store prefix_checkpoints/reddit --print_every 1000 --eval_every 10000 --file embed_store/reddit50_train_embed.pth --num_token 50 --bs 5

deactivate
```
After training, rename the best performing checkpoints as 'final_checkpoint.pth'. The best performing checkpoints produce the higest averge ROUGE scores.
<hr>

## DAPA
Reproduce DAPA results by running the following:

```
source env2/bin/activate
mkdir DAPA

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --diag_file embed_store/scientific_data_AIC_test_10_diag.pickle --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/cnn50_test_50_diag_embed.pth --diag_file embed_store/cnn_test_50_diag.pickle --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --file embed_store/reddit50_test_50_diag_embed.pth --diag_file embed_store/reddit_test_50_diag.pickle --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/samsum50_test_50_diag_embed.pth --diag_file embed_store/samsum_test_50_diag.pickle --eval_bs 4

deactivate
```
For the complete set of experiments, refer to runs.sh
