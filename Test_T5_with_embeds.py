from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers

import json
from transformers import Adafactor
import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import random


from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt import PromptForGeneration
from openprompt.data_utils.utils import InputExample

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default=None)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument('--store',type=str,default=None)
parser.add_argument('--file',type=str,default=None)
parser.add_argument('--num_token',type=int,default=None)
args=parser.parse_args()

torch.manual_seed(42)

file=open(args.test_file,'rb')
data_test=pickle.load(file)
file.close()

def read_data(data):        
    lis=[]
    for i in range(len(data)):
        lis.append(InputExample(guid=str(i),text_a=data[i][0],tgt_text=data[i][1]))

    return lis

dataset={}
dataset['validation'] = read_data(data_test)
dataset['name'] = args.test_file.split('.')[0][:-4]

class Train:
    def __init__(self,dataset,args):
        self.dataset = dataset
        self.args=args
        
        self.eval_bs=8
        self.use_cuda = True

        plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
        self.mytemplate = PrefixTuningTemplate(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, file = args.file)

        self.validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=self.mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
            batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
            truncate_method="head")

        self.mytemplate.load_state_dict(torch.load(args.checkpoint))

        self.prompt_model = PromptForGeneration(plm=plm,template=self.mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)                
        
        if self.use_cuda:
            self.prompt_model = self.prompt_model.cuda()
    
        print('testing')
        self.val(0)

    def val(self,epoch):
        generated_sentence = []
        groundtruth_sentence = []
        self.prompt_model.eval()
        
        for step, inputs in enumerate(self.validation_dataloader):
            if self.use_cuda:
                inputs = inputs.cuda()
            
            _,output_sentence=self.prompt_model.generate(inputs,
                                    num_beams=10, \
                                    early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)

            output_sentence=[o.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for o in output_sentence]
            gold = [ii.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for ii in inputs['tgt_text']]
            
            generated_sentence.extend(output_sentence)
            groundtruth_sentence.extend(inputs['tgt_text'])
        print(len(generated_sentence))
        print(len(groundtruth_sentence))

        acc = 0
        file=open(self.args.store+'/'+str(epoch)+self.dataset['name']+'_test_gen.txt','w')
        file1=open(self.args.store+'/'+str(epoch)+self.dataset['name']+'_test_ref.txt','w')
        for i in range(len(generated_sentence)):
            file1.write(groundtruth_sentence[i].strip()+'\n')
            file.write(generated_sentence[i].strip()+'\n')
            if groundtruth_sentence[i].strip() == generated_sentence[i].strip(): acc+=1

        file.close()
        file1.close()
            
        print(100*acc/len(generated_sentence))

trainer=Train(dataset, args)

