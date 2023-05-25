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
parser.add_argument('--store',type=str,default=None)
parser.add_argument('--eval_every',type=int,default=1000)
parser.add_argument('--print_every',type=int,default=10000)
parser.add_argument('--bs',type=int,default=5)
parser.add_argument('--back_propogate',type=int,default=5)
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
lens=[len(data_dev1),len(data_dev2),len(data_dev3)]
names = [args.dev_file1.split('.')[0][:-4],args.dev_file2.split('.')[0][:-4],args.dev_file3.split('.')[0][:-4]]
data_dev = data_dev1 + data_dev2 + data_dev3

class Model(nn.Module):
        def __init__(self,model_name):
                super(Model,self).__init__()
                self.model=T5ForConditionalGeneration.from_pretrained(model_name)

        def forward(self,input):
                outputs=self.model(input_ids=input['input_ids'], \
                                           labels=input['labels'], attention_mask=input['attention_mask'],output_hidden_states=True,output_attentions=True)

                return outputs.loss
                

class Train:
        def __init__(self,data,data_val,lens,names, args):
                self.data=data
                self.dev_data=data_val
                self.args=args

                self.lens = lens
                self.names = names

                self.tokenizer=T5Tokenizer.from_pretrained(args.model_name)
                self.model=nn.DataParallel(Model(args.model_name),device_ids=[args.device])
                self.model.to(f'cuda:{self.model.device_ids[0]}')  
               
                #self.optimizer=optim.AdamW(self.model.parameters(),lr=0.0015)
                self.optimizer=Adafactor(self.model.parameters(),lr=5e-4,eps=(1e-30, 1e-3),clip_threshold=1.0, \
                beta1=0.0,weight_decay=0.0,relative_step=False, \
                scale_parameter=True,warmup_init=False)

                self.lr_scheduler=transformers. \
                get_polynomial_decay_schedule_with_warmup(self.optimizer, 5000, 30000,power=0.5)

                self.iters=600000
                self.print_every=args.print_every
                self.eval_every=args.eval_every
                self.num_gpus=1
                self.eval_bs=3
                self.bs=args.bs
                self.back_propogate=args.back_propogate
                
                if self.args.test:
                    print('testing')
                    self.val(0)
                else:
                    self.train()

        def generate_batch(self):
                output=random.sample(self.data,self.bs)
                inp,label=[],[]
                for dat in output:
                        inp.append(dat[0])
                        label.append(dat[1])

                return inp,label

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

        def val1(self,o,dev_data,name):
                self.model.eval()
                acc,bs,i=0,self.eval_bs,0
                lis=[]
               
                while i<len(dev_data):
                    bs_=min(bs,len(dev_data)-i)
                    i+=bs_
                    inp,label=[],[]
                    for j in range(i-bs_,i):
                            inp.append(dev_data[j][0])
                            label.append(dev_data[j][1])

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
                            if a1==a2:
                                    acc+=1; #print('ttt')
                

                file=open(self.args.store+'/'+name+'_'+str(o)+'ref.txt','w')
                file1=open(self.args.store+'/'+name+'_'+str(o)+'gen.txt','w')

                for item in lis:
                    file.write(item[1].strip()+'\n')
                    file1.write(item[0].strip()+'\n')
                file1.close()
                file.close()
                return 100*acc/len(self.dev_data)

        def val(self, o):
            data1 = self.dev_data[:self.lens[0]]
            data2 = self.dev_data[self.lens[0]:self.lens[1]+self.lens[0]]
            data3 = self.dev_data[self.lens[1]+self.lens[0]:]

            self.val1(o,data1,self.names[0])
            self.val1(o,data2,self.names[1])
            self.val1(o,data3,self.names[2])

            return 0

        def train(self):

                scalar=0
                for i in range(self.iters):
                        self.model.train()
                        inp,label=self.generate_batch()
                        input=self.preprocess_function(inp,label)
                        loss=self.model(input)

                        scalar+=loss.mean().item()
                        if(i+1)%self.print_every==0:
                                print('iteration={}, training loss={}'.format(i+1,scalar/self.print_every))
                                scalar=0
                        if(i+1)%self.eval_every==0:
                                acc=self.val(i+1)
                                print('validation acc={}'.format(acc))

                                torch.save(self.model.state_dict(),self.args.store+'/'+str(i+1)+'checkpoint.pth')
                        
                        loss/=self.back_propogate
                        loss.mean().backward()
                        if (i+1)%self.back_propogate:
                                self.optimizer.step();
                                self.lr_scheduler.step();
                                self.optimizer.zero_grad()

trainer=Train(data_train,data_dev,lens,names, args)
