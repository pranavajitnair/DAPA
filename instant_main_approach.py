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
parser.add_argument('--eval_bs', type = int, default = None)
args=parser.parse_args()

torch.manual_seed(42)

file=open(args.test_file,'rb')
data_test=pickle.load(file)
file.close()
        
def read_data(data):        
    lis=[]
    for i in range(len(data)):
        lis.append(InputExample(guid=str(i),text_a=data[i][0].replace('<extra_id_','T').replace('>',''),tgt_text=data[i][1].replace('<extra_id_','T').replace('>','')))

    return lis

dataset={}
dataset['test'] = read_data(data_test)
                

class Train:
        def __init__(self,dataset,args):
                self.dataset = dataset
                self.args=args

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

                self.mytemplate = PrefixTuningTemplate(model=plm, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, prefix1 = prefix_state1, prefix2 = prefix_state2, prefix3 = prefix_state3)

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
                self.mytemplate1 = PrefixTuningTemplate1(model=plm1, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})
                self.mytemplate2 = PrefixTuningTemplate1(model=plm2, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})
                self.mytemplate3 = PrefixTuningTemplate1(model=plm3, num_token=args.num_token, tokenizer=tokenizer, placeholder_mapping = {'<text_a>': 'text_a', '<text_b>': 'text_b'})

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

                
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.model.eval()

                print('testing')
                with torch.no_grad(): self.val(0)

        def val1(self, model, inputs):
            model.eval()
            
            if self.use_cuda:
                inputs = inputs.cuda()
            
            _,output_sentence=model.generate(inputs,
                                    num_beams=10, \
                                    early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True)

            output_sentence=[o.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for o in output_sentence]

            return output_sentence


        def val(self,epoch):
            generated_sentence = []
            groundtruth_sentence = []
            self.prompt_model.eval()
            #d1_final, d2_final, d3_final = 0, 0, 0
            
            for step, inputs in enumerate(self.test_dataloader):

                sentences1 = self.val1(self.prompt_model1, inputs)
                sentences2 = self.val1(self.prompt_model2, inputs)
                sentences3 = self.val1(self.prompt_model3, inputs)

                d1, d2, d3 = [], [], []
                for i in range(len(sentences1)):
                    d1.append(F.cosine_similarity(torch.tensor(self.model.encode(sentences1[i])).unsqueeze(0), torch.tensor(self.model.encode(inputs['tgt_text'][i])).unsqueeze(0))[0])
                    d2.append(F.cosine_similarity(torch.tensor(self.model.encode(sentences2[i])).unsqueeze(0), torch.tensor(self.model.encode(inputs['tgt_text'][i])).unsqueeze(0))[0])
                    d3.append(F.cosine_similarity(torch.tensor(self.model.encode(sentences3[i])).unsqueeze(0), torch.tensor(self.model.encode(inputs['tgt_text'][i])).unsqueeze(0))[0])
                
                d1 = torch.tensor(d1).unsqueeze(-1)
                d2 = torch.tensor(d2).unsqueeze(-1)
                d3 = torch.tensor(d3).unsqueeze(-1)

                weights = F.softmax(torch.cat([d1, d2, d3], dim=-1), dim=-1).cuda()
                '''for i in range(len(sentences1)):
                    d1_final += weights[i][0].item()
                    d2_final += weights[i][1].item()
                    d3_final += weights[i][2].item()
                continue'''

                ids = inputs['input_ids']
                tokens = self.tokenizer.batch_decode(ids)
                embeddings = []

                for k in range(len(tokens)):
                    #new = []
                    tokens[k] = tokens[k].replace('<pad>','').replace('</s>','').replace('<unk>','').replace('<s>','').strip()
                    new = self.tokenizer(tokens[k])['input_ids'][1:-1]

                    a = Counter(new)
                    next = [x[0] for x in a.most_common(50)]

                    if len(next) == 0:
                        for i in range(50):
                            next.append(self.tokenizer.convert_tokens_to_ids('<pad>'))

                    elif len(next) < 50:
                        t = []
                        l = 0
                        while True:
                            t.append(next[l%len(next)])
                            l += 1
                            if l == 50: break
                        next = t

                    t=torch.tensor(next).cuda()
                    b = self.plm.shared(t).detach()
                    embeddings.append(b.unsqueeze(0))

                inputs['embeddings'] = torch.cat(embeddings, dim = 0).cuda()

                if self.use_cuda:
                    inputs = inputs.cuda()
                
                _,output_sentence=self.prompt_model.generate(inputs,
                                        num_beams=10, \
                                        early_stopping=True, max_length=200,output_hidden_states=True,output_attentions=True, weights = weights)

                output_sentence=[o.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for o in output_sentence]
                gold = [ii.replace('<unk>','').replace('<pad>','').replace('<s>','').replace('</s>','') for ii in inputs['tgt_text']]
                
                generated_sentence.extend(output_sentence)
                groundtruth_sentence.extend(inputs['tgt_text'])

            #print(d1_final/len(self.test_dataloader), d2_final/len(self.test_dataloader), d3_final/len(self.test_dataloader));return

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
