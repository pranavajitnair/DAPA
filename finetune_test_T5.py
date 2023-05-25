from transformers import T5Tokenizer, T5ForConditionalGeneration, Adafactor
import transformers

import json

import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import random

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default=None)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument('--checkpoint',type=str,default=None)
parser.add_argument('--device',type=int,default=0)
parser.add_argument('--store',type=str,default=None)
parser.add_argument('--eval_bs',type=int,default=6)
args=parser.parse_args()


torch.manual_seed(42)

file=open(args.test_file+'_test.pickle','rb')
data_test=pickle.load(file)
file.close()


class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'], attention_mask=input['attention_mask'],output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Train:
        def __init__(self,data_test,args):
                self.test_data = data_test
                self.args=args

                self.tokenizer=T5Tokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  

                self.eval_bs=args.eval_bs
                
                print('testing')
                self.model.load_state_dict(torch.load(args.checkpoint))
                self.val(0)

        def preprocess_function(self,inputs, targets):
                model_inputs=self.tokenizer(inputs, padding=True, \
                                            return_tensors='pt',max_length=512, truncation=True)
                labels=self.tokenizer(targets,padding=True,max_length=512, truncation=True)

                if True:
                    labels["input_ids"] = [
                        [(l if l != self.tokenizer.pad_token_id else -100) \
                         for l in label] for label in labels["input_ids"]
                    ]
                labels['input_ids']=torch.tensor(labels['input_ids'])
                model_inputs["labels"]=labels["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["input_ids"]=model_inputs["input_ids"].to(f'cuda:{self.model.device_ids[0]}')
                model_inputs["attention_mask"]=model_inputs["attention_mask"].to(f'cuda:{self.model.device_ids[0]}')

                return model_inputs

        def val(self,o):
                self.model.eval()
                acc,bs,i=0,self.eval_bs,0
                lis=[]
               
                while i<len(self.test_data):
                    bs_=min(bs,len(self.test_data)-i)
                    i+=bs_
                    inp,label=[],[]
                    for j in range(i-bs_,i):
                            inp.append(self.test_data[j][0])
                            label.append(self.test_data[j][1])

                    input=self.preprocess_function(inp,label)
                    
                    output=self.model.module.model.generate(input_ids=input['input_ids'],
                                          num_beams=10,attention_mask=input['attention_mask'], \
                                            early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)
                    
                    out=self.tokenizer.batch_decode(output,skip_special_tokens=False)
    
                    for k in range(len(out)):
                            #print(out[k].replace('<pad>','').replace('</s>','').strip())
                            a1=out[k].replace('<pad>','').replace('</s>','').replace('<unk>','').replace('<s>','').strip()
                            a2=label[k].strip()
                            
                            lis.append([a1,a2])                

                file=open(self.args.store+'/'+str(o)+self.args.test_file+'_test_ref.txt','w')
                file1=open(self.args.store+'/'+str(o)+self.args.test_file+'_test_gen.txt','w')

                for item in lis:
                    file.write(item[1].strip()+'\n')
                    file1.write(item[0].strip()+'\n')
                file1.close()
                file.close()

trainer=Train(data_test,args)
