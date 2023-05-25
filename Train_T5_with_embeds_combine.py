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
parser.add_argument('--train_file1',type=str,default=None)
parser.add_argument('--dev_file1',type=str,default=None)
parser.add_argument('--train_file2',type=str,default=None)
parser.add_argument('--dev_file2',type=str,default=None)
parser.add_argument('--train_file3',type=str,default=None)
parser.add_argument('--dev_file3',type=str,default=None)
parser.add_argument('--test',type=bool,default=False)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument('--store',type=str,default=None)
parser.add_argument('--eval_every',type=int,default=1000)
parser.add_argument('--print_every',type=int,default=10000)
parser.add_argument('--bs',type=int,default=5)
parser.add_argument('--file',type=str,default=None)
args=parser.parse_args()

torch.manual_seed(42)

file=open(args.train_file1,'rb')
data_train1=pickle.load(file)
file.close()

file=open(args.dev_file1,'rb')
data_dev1=pickle.load(file)
file.close()

file=open(args.train_file2,'rb')
data_train2=pickle.load(file)
file.close()

file=open(args.dev_file2,'rb')
data_dev2=pickle.load(file)
file.close()

file=open(args.train_file3,'rb')
data_train3=pickle.load(file)
file.close()

file=open(args.dev_file3,'rb')
data_dev3=pickle.load(file)
file.close()

data_train = data_train1 + data_train2 + data_train3
random.shuffle(data_train)

def read_data(data):        
    lis=[]
    for i in range(len(data)):
        lis.append(InputExample(guid=str(i),text_a=data[i][0],tgt_text=data[i][1]))

    return lis

dataset={}
dataset['train'] = read_data(data_train)
dataset['validation1'] = read_data(data_dev1)
dataset['validation2'] = read_data(data_dev2)
dataset['validation3'] = read_data(data_dev3)
names = [args.dev_file1.split('.')[0][:-4],args.dev_file2.split('.')[0][:-4],args.dev_file3.split('.')[0][:-4]]


class Train:
        def __init__(self,dataset,args, names):
                self.dataset = dataset
                self.args=args
                self.names  = names

                self.epochs = 40
                self.print_every=args.print_every
                self.eval_every=args.eval_every
                self.num_gpus=1
                self.eval_bs=10
                self.bs=args.bs
                self.back_propogate=10
                self.use_cuda = True

                plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
                self.mytemplate = PrefixTuningTemplate(model=plm, num_token=50, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, file = args.file)

                self.train_dataloader = PromptDataLoader(dataset=dataset["train"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.bs,shuffle=True, teacher_forcing=True, predict_eos_token=True, # be sure to pass predict_eos_token=True if your tempalte doesn't contain one, or you model may fail to stop generation.
                    truncate_method="head")

                self.validation_dataloader1 = PromptDataLoader(dataset=dataset["validation1"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.validation_dataloader2 = PromptDataLoader(dataset=dataset["validation2"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.validation_dataloader3 = PromptDataLoader(dataset=dataset["validation3"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.prompt_model = PromptForGeneration(plm=plm,template=self.mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)                
               
                if self.use_cuda:
                    self.prompt_model = self.prompt_model.cuda()

                no_decay = ["bias", "LayerNorm.weight"]
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.mytemplate.named_parameters() if (not any(nd in n for nd in no_decay)) and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [p for n, p in self.mytemplate.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                        "weight_decay": 0.0,
                    },
                ]

                #self.optimizer=optim.AdamW(optimizer_grouped_parameters, lr=0.00005)
                self.optimizer=Adafactor(optimizer_grouped_parameters,lr=5e-3,eps=(1e-30, 1e-3),clip_threshold=1.0, \
                beta1=0.0,weight_decay=0.0,relative_step=False, \
                scale_parameter=True,warmup_init=False)
                #self.lr_scheduler=transformers.get_polynomial_decay_schedule_with_warmup(self.optimizer, 5000, 30000, power = 0.5)
                
                if self.args.test:
                    self.val(0)
                else:
                    self.train()

        def val1(self,data,epoch,name):
            generated_sentence = []
            groundtruth_sentence = []
            self.prompt_model.eval()
            
            for step, inputs in enumerate(data):
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
            file=open(self.args.store+'/'+name+'_'+str(epoch)+'gen.txt','w')
            file1=open(self.args.store+'/'+name+'_'+str(epoch)+'ref.txt','w')
            for i in range(len(generated_sentence)):
                file1.write(groundtruth_sentence[i].strip()+'\n')
                file.write(generated_sentence[i].strip()+'\n')
                if groundtruth_sentence[i].strip() == generated_sentence[i].strip(): acc+=1

            file.close()
            file1.close()
                
            print(100*acc/len(generated_sentence))

        def val(self, o):
            self.val1(self.validation_dataloader1,o,self.names[0])
            self.val1(self.validation_dataloader2,o,self.names[1])
            self.val1(self.validation_dataloader3,o,self.names[2])

        def train(self):
            global_step = 0
            tot_loss = 0
            log_loss = 0
            for epoch in range(self.epochs):
                self.prompt_model.train()
                for step, inputs in enumerate(self.train_dataloader):
                    global_step +=1
                    if self.use_cuda:
                        inputs = inputs.cuda()
                    loss = self.prompt_model(inputs)
                    #loss += 0.1*(self.mytemplate.loss + self.mytemplate.loss1)
                    loss.backward()
                    tot_loss += loss.item()
                    torch.nn.utils.clip_grad_norm_(self.mytemplate.parameters(), 1.0)
                    self.optimizer.step()
                    #self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    if (global_step) %self.print_every ==0:
                        print("Epoch {}, global_step {} average loss: {}".format(epoch, global_step, (tot_loss-log_loss)/self.print_every), flush=True)
                        #print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
                        log_loss = tot_loss
                    if (global_step) %self.eval_every ==0:
                        self.val(global_step)
                        torch.save(self.mytemplate.state_dict(),self.args.store+'/'+str(global_step)+'checkpoint.pth')

trainer=Train(dataset, args, names)
