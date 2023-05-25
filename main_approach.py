from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
from transformers import RobertaTokenizer, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer

import json
from transformers import Adafactor
import torch
import torch.optim as optim
import pickle
import torch.nn as nn
import random
import torch.nn.functional as F
from collections import Counter

from openprompt.plms import load_plm
from openprompt import PromptDataLoader
from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate
from openprompt.prompts.prefix_tuning_template_env4 import PrefixTuningTemplate1
from openprompt.prompts.prefix_state import PrefixState
from openprompt import PromptForGeneration, PromptForGeneration1
from openprompt.data_utils.utils import InputExample

import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--test_file',type=str,default=None)
parser.add_argument('--model_name',type=str,default='t5-base')
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument('--store',type=str,default=None)
parser.add_argument('--num_token', type = int, default = None)
parser.add_argument('--file1',type=str,default=None)
parser.add_argument('--file2',type=str,default=None)
parser.add_argument('--file3',type=str,default=None)
parser.add_argument('--file', type = str, default = None)
parser.add_argument('--diag_file', type = str, default = None)
parser.add_argument('--eval_bs', type = int, default = None)
parser.add_argument('--num_diag', type = int, default = 50)
parser.add_argument('--average', type = bool, default = False)
parser.add_argument('--maximum', type = bool, default =False)
parser.add_argument('--use_wte', type = bool, default = False)
parser.add_argument('--average_later', type = bool, default = False)
args=parser.parse_args()

torch.manual_seed(42)

file=open(args.test_file,'rb')
data_test=pickle.load(file)
file.close()

file=open(args.diag_file,'rb')
data_diag=pickle.load(file)[:args.num_diag]
file.close()
        
def read_data(data):        
    lis=[]
    for i in range(len(data)):
        lis.append(InputExample(guid=str(i),text_a=data[i][0].replace('<extra_id_','T').replace('>',''),tgt_text=data[i][1].replace('<extra_id_','T').replace('>','')))

    return lis

dataset={}
dataset['test'] = read_data(data_test)
dataset['diag'] = read_data(data_diag)
                

class Train:
        def __init__(self,dataset,args):
                self.dataset = dataset
                self.args=args

                print(args.use_wte)
                print(args.average)
                print(args.maximum)

                self.embed_params = torch.load(args.file).cuda()
                #self.weights = torch.load(args.weights).cuda()

                self.name = self.args.test_file.split('_')[0]

                self.eval_bs=args.eval_bs
                self.use_cuda = True

                plm, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
                self.tokenizer = tokenizer
                self.plm = plm
                prefix_state1 = PrefixState(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})
                prefix_state2 = PrefixState(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})
                prefix_state3 = PrefixState(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})
                prefix_state1.load_state_dict(torch.load(args.file1))
                prefix_state2.load_state_dict(torch.load(args.file2))
                prefix_state3.load_state_dict(torch.load(args.file3))

                self.mytemplate = PrefixTuningTemplate(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, prefix1 = prefix_state1, prefix2 = prefix_state2, prefix3 = prefix_state3, maximum = args.maximum, use_wte = args.use_wte)

                self.test_dataloader = PromptDataLoader(dataset=dataset["test"], template=self.mytemplate, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.prompt_model = PromptForGeneration(plm=plm,template=self.mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)                
               
                if self.use_cuda:
                    self.prompt_model = self.prompt_model.cuda()

                plm1, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
                plm2, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
                plm3, tokenizer, model_config, WrapperClass = load_plm(args.model_name.split('-')[0], args.model_name)
                self.mytemplate1 = PrefixTuningTemplate1(model=plm1, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, file = args.file)
                self.mytemplate2 = PrefixTuningTemplate1(model=plm2, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, file = args.file)
                self.mytemplate3 = PrefixTuningTemplate1(model=plm3, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, file = args.file)

                self.validation_dataloader1 = PromptDataLoader(dataset=dataset["diag"], template=self.mytemplate1, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")
                
                self.validation_dataloader2 = PromptDataLoader(dataset=dataset["diag"], template=self.mytemplate2, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.validation_dataloader3 = PromptDataLoader(dataset=dataset["diag"], template=self.mytemplate3, tokenizer=tokenizer,
                    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=200,
                    batch_size=self.eval_bs,shuffle=False, teacher_forcing=False, predict_eos_token=True,
                    truncate_method="head")

                self.mytemplate1.load_state_dict(torch.load(args.file1))
                self.mytemplate2.load_state_dict(torch.load(args.file2))
                self.mytemplate3.load_state_dict(torch.load(args.file3))

                self.prompt_model1 = PromptForGeneration1(plm=plm1,template=self.mytemplate1, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
                self.prompt_model2 = PromptForGeneration1(plm=plm2,template=self.mytemplate2, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
                self.prompt_model3 = PromptForGeneration1(plm=plm3,template=self.mytemplate3, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)                
               
                if self.use_cuda:
                    self.prompt_model1 = self.prompt_model1.cuda()
                    self.prompt_model2 = self.prompt_model2.cuda()
                    self.prompt_model3 = self.prompt_model3.cuda()

                sentences1 = self.val1(self.prompt_model1, self.validation_dataloader1)
                sentences2 = self.val1(self.prompt_model2, self.validation_dataloader2)
                sentences3 = self.val1(self.prompt_model3, self.validation_dataloader3)
                
                model = SentenceTransformer('all-MiniLM-L6-v2')
                d1, d2, d3 = 0, 0, 0
                if args.average_later == False:
                    for i in range(len(sentences1)):
                        #print(torch.tensor(model.encode(sentences1[i])).shape, torch.tensor(model.encode(dataset['validation'][i].text_a)).shape)
                        d1 += F.cosine_similarity(torch.tensor(model.encode(sentences1[i])).unsqueeze(0), torch.tensor(model.encode(dataset['diag'][i].text_a)).unsqueeze(0))[0]
                        d2 += F.cosine_similarity(torch.tensor(model.encode(sentences2[i])).unsqueeze(0), torch.tensor(model.encode(dataset['diag'][i].text_a)).unsqueeze(0))[0]
                        d3 += F.cosine_similarity(torch.tensor(model.encode(sentences3[i])).unsqueeze(0), torch.tensor(model.encode(dataset['diag'][i].text_a)).unsqueeze(0))[0]
                
                        self.weights = F.softmax(torch.tensor([d1, d2, d3]), dim = -1).cuda()

                else:
                    self.weights = 0
                    for i in range(len(sentences1)):
                        #print(torch.tensor(model.encode(sentences1[i])).shape, torch.tensor(model.encode(dataset['validation'][i].text_a)).shape)
                        d1 = F.cosine_similarity(torch.tensor(model.encode(sentences1[i])).unsqueeze(0), torch.tensor(model.encode(dataset['diag'][i].text_a)).unsqueeze(0))[0]
                        d2 = F.cosine_similarity(torch.tensor(model.encode(sentences2[i])).unsqueeze(0), torch.tensor(model.encode(dataset['diag'][i].text_a)).unsqueeze(0))[0]
                        d3 = F.cosine_similarity(torch.tensor(model.encode(sentences3[i])).unsqueeze(0), torch.tensor(model.encode(dataset['diag'][i].text_a)).unsqueeze(0))[0]
                        self.weights = self.weights + F.softmax(torch.tensor([d1, d2, d3]), dim = -1).cuda()
                    
                    self.weights = self.weights/len(sentences1)

                if args.average == True:
                    self.weights = torch.tensor([1/3, 1/3, 1/3]).cuda()

                print('testing')
                #print(self.weights)
                with torch.no_grad(): self.val(0)

        def val1(self, model, data):
            generated_sentence = []
            groundtruth_sentence = []
            model.eval()
            
            for step, inputs in enumerate(data):
                if self.use_cuda:
                    inputs = inputs.cuda()
                
                _,output_sentence=model.generate(inputs,
                                        num_beams=10, \
                                        early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)

                output_sentence=[o.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for o in output_sentence]
                gold = [ii.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for ii in inputs['tgt_text']]
                
                generated_sentence.extend(output_sentence)
                groundtruth_sentence.extend(inputs['tgt_text'])

            return generated_sentence


        def val(self,epoch):
            generated_sentence = []
            groundtruth_sentence = []
            self.prompt_model.eval()
            
            for step, inputs in enumerate(self.test_dataloader):
                ids = inputs['input_ids']
                tokens = self.tokenizer.batch_decode(ids)

                inputs['embeddings'] = self.embed_params.unsqueeze(0).tile(len(tokens), 1, 1)

                if self.use_cuda:
                    inputs = inputs.cuda()
                
                _,output_sentence=self.prompt_model.generate(inputs,
                                        num_beams=10, \
                                        early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True, weights = self.weights.unsqueeze(0).tile(len(tokens), 1))

                output_sentence=[o.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for o in output_sentence]
                gold = [ii.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for ii in inputs['tgt_text']]
                
                generated_sentence.extend(output_sentence)
                groundtruth_sentence.extend(inputs['tgt_text'])
            print(len(generated_sentence))
            print(len(groundtruth_sentence))

            acc = 0
            file=open(self.args.store+'/'+str(epoch)+self.name+'_test_gen.txt','w')
            file1=open(self.args.store+'/'+str(epoch)+self.name+'_test_ref.txt','w')
            for i in range(len(generated_sentence)):
                file1.write(groundtruth_sentence[i].strip()+'\n')
                file.write(generated_sentence[i].strip()+'\n')
                if groundtruth_sentence[i].strip() == generated_sentence[i].strip(): acc+=1

            file.close()
            file1.close()
                
            print(100*acc/len(generated_sentence))

trainer=Train(dataset,args)
