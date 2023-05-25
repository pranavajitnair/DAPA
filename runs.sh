# Create the following three virtual environments

virtualenv --python=python3 env         # for Prefix Tuning experiments
source env/bin/activate
cd OpenPrompt4
python setup.py install
cd ../transformers
pip install -e ./
deactivate

virtualenv --python=python3 env1         # for Finetuning experiments
source env1/bin/activate
cd transformers1
pip install -e ./
deactivate

virtualenv --python=python3 env2         # for DAPA experiments
source env2/bin/activate
cd OpenPrompt8
python setup.py install
cd ../transformers
pip install -e ./
deactivate


# Create checkpoints for DAPA

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

# After training, rename the best performing checkpoints as 'final_checkpoint.pth'

# ERM-prefix

source env/bin/activate
mkdir combine_prefix
cd combine_prefix
mkdir no_scientific
mkdir no_cnn
mkdir no_reddit
mkdir no_samsum
cd ..

# Training

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds_combine.py --train_file1 cnn_train.pickle --dev_file1 cnn_dev.pickle -train_file2 scientific_data_AIC_train.pickle --dev_file2 scientific_data_AIC_dev.pickle -train_file3 samsum_train.pickle --dev_file3 samsum_dev.pickle --model_name t5-small --store combine_prefix/no_reddit --print_every 1000 --eval_every 10000 --file embed_store/leave_one_out_embed/no_reddit_50_embed.pth --num_token 50 --bs 10

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds_combine.py --train_file1 cnn_train.pickle --dev_file1 cnn_dev.pickle -train_file2 reddit_train.pickle --dev_file2 reddit_dev.pickle -train_file3 samsum_train.pickle --dev_file3 samsum_dev.pickle --model_name t5-small --store combine_prefix/no_scientific --print_every 1000 --eval_every 10000 --file embed_store/leave_one_out_embed/no_scientific_50_embed.pth --num_token 50 --bs 10

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds_combine.py --train_file1 cnn_train.pickle --dev_file1 cnn_dev.pickle -train_file2 scientific_data_AIC_train.pickle --dev_file2 scientific_data_AIC_dev.pickle -train_file3 reddit_train.pickle --dev_file3 reddit_dev.pickle --model_name t5-small --store combine_prefix/no_samsum --print_every 1000 --eval_every 10000 --file embed_store/leave_one_out_embed/no_samsum_50_embed.pth --num_token 50 --bs 10

CUDA_VISIBLE_DEVICES=0 python Train_T5_with_embeds_combine.py --train_file1 samsum_train.pickle --dev_file1 samsum_dev.pickle -train_file2 scientific_data_AIC_train.pickle --dev_file2 scientific_data_AIC_dev.pickle -train_file3 reddit_train.pickle --dev_file3 reddit_dev.pickle --model_name t5-small --store combine_prefix/no_cnn --print_every 1000 --eval_every 5000 --file embed_store/leave_one_out_embed/no_cnn_50_embed.pth --num_token 50 --bs 5

# Testing (Rename the best performing checkpoint to final_checkpoint.pth)

CUDA_VISIBLE_DEVICES=0 python Test_T5_with_embeds.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --checkpoint combine_prefix/no_scientific/final_checkpoint.pth --store combine_prefix/no_scientific --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --num_token 50

CUDA_VISIBLE_DEVICES=0 python Test_T5_with_embeds.py --test_file samsum_test.pickle --model_name t5-small --checkpoint combine_prefix/no_samsum/final_checkpoint.pth --store combine_prefix/no_samsum --file embed_store/samsum50_test_50_diag_embed.pth --num_token 50

CUDA_VISIBLE_DEVICES=0 python Test_T5_with_embeds.py --test_file cnn_test.pickle --model_name t5-small --checkpoint combine_prefix/no_cnn/final_checkpoint.pth --store combine_prefix/no_cnn --file embed_store/cnn50_test_50_diag_embed.pth --num_token 50

CUDA_VISIBLE_DEVICES=0 python Test_T5_with_embeds.py --test_file reddit_test.pickle --model_name t5-small --checkpoint combine_prefix/no_reddit/final_checkpoint.pth --store combine_prefix/no_reddit --file embed_store/reddit50_test_50_diag_embed.pth --num_token 50

deactivate

# ERM-finetune

source env1/bin/activate
mkdir combine_finetune
ccd combine_finetune
mkdir no_scientific
mkdir no_cnn
mkdir no_reddit
mkdir no_samsum
cd ..


# Training

CUDA_VISIBLE_DEVICES=0 python finetune_multiple_T5.py --train_file1 cnn_train.pickle --dev_file1 cnn_dev.pickle -train_file2 scientific_data_AIC_train.pickle --dev_file2 scientific_data_AIC_dev.pickle -train_file3 samsum_train.pickle --dev_file3 samsum_dev.pickle --model_name t5-small --store combine_finetune/no_reddit --print_every 1000 --eval_every 10000

CUDA_VISIBLE_DEVICES=0 python finetune_multiple_T5.py --train_file1 cnn_train.pickle --dev_file1 cnn_dev.pickle -train_file2 reddit_train.pickle --dev_file2 reddit_dev.pickle -train_file3 samsum_train.pickle --dev_file3 samsum_dev.pickle --model_name t5-small --store combine_finetune/no_scientific --print_every 1000 --eval_every 10000

CUDA_VISIBLE_DEVICES=0 python finetune_multiple_T5.py --train_file1 cnn_train.pickle --dev_file1 cnn_dev.pickle -train_file2 scientific_data_AIC_train.pickle --dev_file2 scientific_data_AIC_dev.pickle -train_file3 reddit_train.pickle --dev_file3 reddit_dev.pickle --model_name t5-small --store combine_finetune/no_samsum --print_every 1000 --eval_every 10000

CUDA_VISIBLE_DEVICES=0 python finetune_multiple_T5.py --train_file1 samsum_train.pickle --dev_file1 samsum_dev.pickle -train_file2 scientific_data_AIC_train.pickle --dev_file2 scientific_data_AIC_dev.pickle -train_file3 reddit_train.pickle --dev_file3 reddit_dev.pickle --model_name t5-small --store combine_finetune/no_cnn --print_every 1000 --eval_every 5000

# Testing (Rename the best performing checkpoint to final_checkpoint.pth)

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file scientific_data_AIC --model_name t5-small --store combine_finetune/no_scientific --checkpoint combine_finetune/no_scientific/final_checkpoint.pth --eval_bs 3

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file samsum --model_name t5-small --store combine_finetune/no_sasum --checkpoint combine_finetune/no_samsum/final_checkpoint.pth --eval_bs 3

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file reddit --model_name t5-small --store combine_finetune/no_reddit --checkpoint combine_finetune/no_reddit/final_checkpoint.pth --eval_bs 3

CUDA_VISIBLE_DEVICES=0 python finetune_test_T5.py --test_file cnn --model_name t5-small --store combine_finetune/no_cnn --checkpoint combine_finetune/no_cnn/final_checkpoint.pth --eval_bs 3

deactivate

# DAPA and its variants

source env2/bin/activate
mkdir DAPA
mkdir DAPA-alt
mkdir DAPA-embed
mkdir DAPA-average
mkdir DAPA-max
mkdir DAPA-inst

DAPA

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --diag_file embed_store/scientific_data_AIC_test_10_diag.pickle --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/cnn50_test_50_diag_embed.pth --diag_file embed_store/cnn_test_50_diag.pickle --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --file embed_store/reddit50_test_50_diag_embed.pth --diag_file embed_store/reddit_test_50_diag.pickle --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/samsum50_test_50_diag_embed.pth --diag_file embed_store/samsum_test_50_diag.pickle --eval_bs 4

DAPA-average

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA-average --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --diag_file embed_store/scientific_data_AIC_test_10_diag.pickle --eval_bs 4 --average True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA-average --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/cnn50_test_50_diag_embed.pth --diag_file embed_store/cnn_test_50_diag.pickle --eval_bs 4 --average True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA-average --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --file embed_store/reddit50_test_50_diag_embed.pth --diag_file embed_store/reddit_test_50_diag.pickle --eval_bs 4 --average True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA-average --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/samsum50_test_50_diag_embed.pth --diag_file embed_store/samsum_test_50_diag.pickle --eval_bs 4 --average True

DAPA-max

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA-max --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --diag_file embed_store/scientific_data_AIC_test_10_diag.pickle --eval_bs 4 --maximum True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA-max --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/cnn50_test_50_diag_embed.pth --diag_file embed_store/cnn_test_50_diag.pickle --eval_bs 4 --maximum True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA-max --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --file embed_store/reddit50_test_50_diag_embed.pth --diag_file embed_store/reddit_test_50_diag.pickle --eval_bs 4 --maximum True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA-max --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/samsum50_test_50_diag_embed.pth --diag_file embed_store/samsum_test_50_diag.pickle --eval_bs 4 --maximum True

DAPA-alt

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA-alt --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --diag_file embed_store/scientific_data_AIC_test_10_diag.pickle --eval_bs 4 --average_later True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA-alt --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/cnn50_test_50_diag_embed.pth --diag_file embed_store/cnn_test_50_diag.pickle --eval_bs 4 --average_later True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA-alt --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --file embed_store/reddit50_test_50_diag_embed.pth --diag_file embed_store/reddit_test_50_diag.pickle --eval_bs 4 --average_later True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA-alt --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/samsum50_test_50_diag_embed.pth --diag_file embed_store/samsum_test_50_diag.pickle --eval_bs 4 --average_later True

DAPA-embed

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA-embed --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/scientific_data_AIC50_test_10_diag_embed.pth --diag_file embed_store/scientific_data_AIC_test_10_diag.pickle --eval_bs 4 --use_wte True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA-embed --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/cnn50_test_50_diag_embed.pth --diag_file embed_store/cnn_test_50_diag.pickle --eval_bs 4 --use_wte True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA-embed --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --file embed_store/reddit50_test_50_diag_embed.pth --diag_file embed_store/reddit_test_50_diag.pickle --eval_bs 4 --use_wte True

CUDA_VISIBLE_DEVICES=0 python main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA-embed --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --file embed_store/samsum50_test_50_diag_embed.pth --diag_file embed_store/samsum_test_50_diag.pickle --eval_bs 4 --use_wte True

DAPA-inst

CUDA_VISIBLE_DEVICES=0 python instant_main_approach.py --test_file scientific_data_AIC_test.pickle --model_name t5-small --store DAPA-inst --num_token 50 --file1 prefix_checkpoints/samsum/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python instant_main_approach.py --test_file cnn_test.pickle --model_name t5-small --store DAPA-inst --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/samsum/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python instant_main_approach.py --test_file reddit_test.pickle --model_name t5-small --store DAPA-inst --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/samsum/final_checkpoint.pth --eval_bs 4

CUDA_VISIBLE_DEVICES=0 python instant_main_approach.py --test_file samsum_test.pickle --model_name t5-small --store DAPA-inst --num_token 50 --file1 prefix_checkpoints/scientific/final_checkpoint.pth --file2 prefix_checkpoints/cnn/final_checkpoint.pth --file3 prefix_checkpoints/reddit/final_checkpoint.pth --eval_bs 4

deactivate
