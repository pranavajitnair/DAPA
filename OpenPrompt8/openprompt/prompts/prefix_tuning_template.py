from functools import partial
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from openprompt.data_utils import InputFeatures
import os
import torch
from torch import nn
from typing import *
from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts.prefix_state import PrefixState
from openprompt.utils.logging import logger


class PrefixTuningTemplate(Template):
    r"""This is the implementation which support T5 and other Encoder-Decoder model,
    as soon as their blocks allows the ``past_key_values`` to be injected to the model.
    This implementation modifies the huggingface's T5 forward without touching the code-base.
    However, it may fail to work when used in DataParallel model. Please use it using
    single gpu or model-parallel training.
    Args:
        model (:obj:`PreTrainedModel`): The pre-trained model.
        plm_config (:obj:`PretrainedConfig`): The configuration of the current pre-trained model.
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model.
        mapping_hook (:obj:`nn.Module`, optional):
        text (:obj:`str`, optional):
        num_token (:obj:`int`, optional):
        placeholder_mapping (:obj:`dict`):
        prefix_dropout (:obj:`float`, optional): The dropout rate for the prefix sequence.
    """
    registered_inputflag_names = ["loss_ids", 'shortenable_ids']

    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 mapping_hook: Optional[nn.Module] = None,
                 text: Optional[str] = None,
                 num_token: Optional[int] = 5,
                 placeholder_mapping: dict = {'<text_a>':'text_a', '<text_b>':'text_b'},
                 prefix_dropout: Optional[float] = 0.0,
                 mid_dim: Optional[int] =  512,
                 using_encoder_past_key_values: Optional[bool] = True,
                 using_decoder_past_key_values: Optional[bool] = True,
                 using_cross: Optional[bool] = True,
                 prefix1: Optional[PrefixState] = None,
                 prefix2: Optional[PrefixState] = None,
                 prefix3: Optional[PrefixState] = None,
                 maximum: Optional[bool] = False,
                 use_wte: Optional[bool] = False
                ):
        super().__init__(tokenizer=tokenizer,
                         placeholder_mapping=placeholder_mapping)
        raw_embedding = model.get_input_embeddings()
        self.config = model.config
        self.mapping_hook = mapping_hook
        self.embedding_size = raw_embedding.weight.shape[-1]
        self.num_token = num_token

        self.weights = 0
        self.use_wte = use_wte
        self.maximum = maximum

        self.prefix1 = prefix1
        self.prefix2 = prefix2
        self.prefix3 = prefix3
        #self.prefix4 = prefix4

        for n,p in self.prefix1.named_parameters():
            p.requires_grad = False
        for n,p in self.prefix2.named_parameters():
            p.requires_grad = False
        for n,p in self.prefix3.named_parameters():
            p.requires_grad = False
        #for n,p in self.prefix4.named_parameters():
        #    p.requires_grad = False

        self.using_cross = using_cross
        self.using_encoder_past_key_values = using_encoder_past_key_values
        self.using_decoder_past_key_values = using_decoder_past_key_values
        assert (self.using_encoder_past_key_values or self.using_decoder_past_key_values), "Can't be both False."
        if not self.config.is_encoder_decoder and not self.using_decoder_past_key_values:
            logger.warning("Ignore using_decoder_past_key_values=False in a decoder-only LM.")

        if isinstance(self.config, T5Config):
            self.n_layer = self.config.num_layers
            self.n_embd = self.config.d_model
            self.n_head = self.config.num_heads
            self.n_decoder_layer = self.config.num_decoder_layers
            self.match_n_decoder_layer = self.n_decoder_layer
            self.match_n_layer = self.n_layer
        elif isinstance(self.config, GPT2Config):
            self.n_decoder_layer = self.config.n_layer
            self.n_embd = self.config.n_embd
            self.n_head = self.config.n_head
            self.match_n_decoder_layer = self.n_decoder_layer
        self.mid_dim = mid_dim
        self.match_n_head = self.n_head
        self.match_n_embd = self.n_embd // self.n_head
        self.prefix_dropout = prefix_dropout
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.default_text1 = '{"placeholder": "text_a"} {"mask"}'
        self.default_text2 = '{"placeholder": "text_a"} {"placeholder": "text_b"} {"mask"}'

        self.text = text

        self.generate_parameters() # in prefix tuning the template text has no interact with the parameters.

        self.plm_modified = False # flag to indicate whether the function of plm are replaced for prefix tuning.
        self.modify_plm(model)


    def on_text_set(self):
        self.text = self.parse_text(self.text)
        self.generate_parameters()


    def get_past_key_values(self, prefix, embeddings, batch_size=1):
        pvs = []
        self.embed=nn.Embedding.from_pretrained(embeddings)
        if prefix.config.is_encoder_decoder and prefix.using_encoder_past_key_values:

            input_tokens = prefix.input_tokens.unsqueeze(0).expand(batch_size, -1)
            if self.use_wte == False:
                temp_control = self.embed(input_tokens)
            else:
                temp_control = prefix.wte(input_tokens)
            past_key_values = prefix.control_trans(temp_control) #bsz, seqlen, layer*emb
            _, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(batch_size, seqlen, prefix.match_n_layer * 2, prefix.match_n_head,
                                                prefix.match_n_embd)
            past_key_values = prefix.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]) #.split(2)
            pvs.append(past_key_values)
        else:
            pvs.append(None)

        if (not prefix.config.is_encoder_decoder) or prefix.using_decoder_past_key_values:

            decoder_input_tokens = prefix.input_tokens.unsqueeze(0).expand(batch_size, -1)
            if self.use_wte == False:
                decoder_temp_control = self.embed(decoder_input_tokens)
            else:
                decoder_temp_control = prefix.decoder_wte(input_tokens)
            decoder_past_key_values = prefix.decoder_control_trans(decoder_temp_control) #bsz, seqlen, layer*emb
            _, decoder_seqlen, _ = decoder_past_key_values.shape
            decoder_past_key_values = decoder_past_key_values.view(batch_size, decoder_seqlen, prefix.match_n_decoder_layer * 2, prefix.match_n_head,
                                                prefix.match_n_embd)
            decoder_past_key_values = prefix.dropout(decoder_past_key_values)
            decoder_past_key_values = decoder_past_key_values.permute([2, 0, 3, 1, 4]) #.split(2)
            pvs.append(decoder_past_key_values)

        
        if  (not prefix.config.is_encoder_decoder) or prefix.using_cross:

            cross_input_tokens = prefix.input_tokens.unsqueeze(0).expand(batch_size, -1)
            if self.use_wte == False: 
                cross_temp_control = self.embed(cross_input_tokens)
            else:
                cross_temp_control = prefix.cross_wte(input_tokens)
            cross_past_key_values = prefix.cross_control_trans(cross_temp_control) #bsz, seqlen, layer*emb
            _, cross_seqlen, _ = cross_past_key_values.shape
            cross_past_key_values = cross_past_key_values.view(batch_size, cross_seqlen, prefix.match_n_decoder_layer * 2, prefix.match_n_head,
                                                prefix.match_n_embd)
            cross_past_key_values = prefix.dropout(cross_past_key_values)
            cross_past_key_values = cross_past_key_values.permute([2, 0, 3, 1, 4]) #.split(2)
            pvs.append(cross_past_key_values)


        else:
            pvs.append(None)
        return pvs

    def get_past_key_values1(self, embeddings, batch_size = 1):
        x = embeddings.shape[0]
        temp_enc = []
        temp_dec = []
        temp_cross = []

        for i in range(x):

            prefix11 = self.get_past_key_values(self.prefix1, embeddings[i])
            prefix22 = self.get_past_key_values(self.prefix2, embeddings[i])
            prefix33 = self.get_past_key_values(self.prefix3, embeddings[i])
            
            pvs = []

            enc1, enc2, enc3 = prefix11[0], prefix22[0], prefix33[0]
            dec1, dec2, dec3 = prefix11[1], prefix22[1], prefix33[1]
            cross1, cross2, cross3 = prefix11[2], prefix22[2], prefix33[2]

            enc = torch.cat([enc1.unsqueeze(-1), enc2.unsqueeze(-1), enc3.unsqueeze(-1)], dim = -1) #, enc4.unsqueeze(-1)], dim = -1)
            dec = torch.cat([dec1.unsqueeze(-1), dec2.unsqueeze(-1), dec3.unsqueeze(-1)], dim = -1) #,  dec4.unsqueeze(-1)], dim = -1)
            cross = torch.cat([cross1.unsqueeze(-1), cross2.unsqueeze(-1), cross3.unsqueeze(-1)], dim = -1) #, cross4.unsqueeze(-1)], dim = -1)
            
            #weight = self.weights[i].unsqueeze(0).unsqueeze(0).unsqueeze(2).unsqueeze(2).unsqueeze(2)
            #weight = torch.tensor([9.9940e-01, 6.0219e-04, 8.9582e-07]).cuda()
            weight = self.weights[i].unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            #print(weight.shape,enc.shape)
            #print(weight.view(-1))
            if self.maximum ==  False:
                enc = enc* weight
                dec = dec* weight
                cross = cross* weight

            else:
                enc = torch.max(enc,dim=-1).values
                dec = torch.max(dec,dim=-1).values
                cross = torch.max(cross,dim=-1).values

            enc = torch.sum(enc,dim=-1)
            dec = torch.sum(dec,dim=-1)
            cross = torch.sum(cross,dim=-1)
            #print(enc.shape)

            temp_cross.append(cross)
            temp_enc.append(enc)
            temp_dec.append(dec)

        enc = torch.cat(temp_enc, dim=1)
        dec = torch.cat(temp_dec, dim=1)
        cross = torch.cat(temp_cross, dim=1)

        pvs.append(enc.split(2))
        pvs.append(dec.split(2))
        pvs.append(cross.split(2))
        
        return pvs

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        return
        # self.embed=nn.Embedding.from_pretrained(torch.load(self.file))
    
    def wrap_one_example(self, example) -> List[Dict]:
        if self.text is None:
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def expand_to_batchsize(self, tup,  batch_size):
        return tuple(t.expand(-1, batch_size,-1,-1,-1) for t in tup)

    def expand_to_batchsize_for_layer(self, tup,  batch_size, layer_id):
        return tup[layer_id].tile(1, batch_size//tup[layer_id].shape[1],1,1,1)



    def process_batch(self, batch: Union[Dict, InputFeatures], embeddings) -> Union[Dict, InputFeatures]:
        r"""
        Convert input_ids to inputs_embeds
        for normal token, use the embedding inside PLM
        for new token, use MLP or LSTM
        """
        batch_size = batch['input_ids'].size(0)
        self.past_key_values = self.get_past_key_values1(embeddings)
        #print(self.past_key_values[0][0].shape)

        if self.config.is_encoder_decoder:
            # the attention_mask is encoder attention mask, the new token mask will be added in modified_encoder_forward.
            pass
        else: # the attention_mask is decoder attention mask
            past_key_values = self.expand_to_batchsize(self.past_key_values[1], batch_size)
            if 'attention_mask' in batch:
                am = batch['attention_mask']
                batch['attention_mask'] = torch.cat([torch.ones((batch_size,self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
            batch['past_key_values'] = past_key_values
        return batch

    def modify_plm(self, model):
        if self.plm_modified:
            return None
        if isinstance(model, T5ForConditionalGeneration):
            if self.using_encoder_past_key_values:
                backup_encoder_forward_functions = []
                for i, layer_module in enumerate(model.encoder.block):
                    backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                    def modified_encoder_forward(*args, **kwargs):
                        layer_id = kwargs.pop('layer_id')
                        batch_size = args[0].shape[0]
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] =self.expand_to_batchsize_for_layer(self.past_key_values[0], batch_size, layer_id).to(device)
                        if kwargs['attention_mask'] is not None:
                            am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
                            kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                        return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)

            if self.using_decoder_past_key_values:
                backup_decoder_self_attn_forward_functions = []
                for i, layer_module in enumerate(model.decoder.block):
                    backup_decoder_self_attn_forward_functions.append(layer_module.layer[0].forward)
                    def modified_decoder_self_attn_forward(*args, **kwargs):
                        batch_size = args[0].shape[0]
                        #print(batch_size)
                        layer_id = kwargs.pop('layer_id')
                        device = args[0].device
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.past_key_values[1], batch_size, layer_id).to(device)
                        if kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1):
                            pass
                        elif kwargs['past_key_value'][0].size(-2) + args[0].size(-2) == kwargs['attention_mask'].size(-1) +self.num_token:
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat([torch.zeros((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                        else:
                            raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                                attention mask length {}".format(kwargs['past_key_value'][0].size(-2),
                                args[0].size(-2),kwargs['attention_mask'].size(-1)))

                        return backup_decoder_self_attn_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[0].forward = partial(modified_decoder_self_attn_forward, layer_id=i)

            if True:
                backup_cross_self_attn_forward_functions = []
                for i, layer_module in enumerate(model.decoder.block):
                    # print(layer_module.layer[1])
                    backup_cross_self_attn_forward_functions.append(layer_module.layer[1].forward)
                    def modified_cross_self_attn_forward(*args, **kwargs):
                        # print('len',len(args),len(kwargs))
                        # for keys in kwargs: print(keys)
                        batch_size = args[0].shape[0]
                        layer_id = kwargs.pop('layer_id')
                        device = args[0].device
                        # print(len(kwargs))
                        # print(args[0].shape,kwargs['attention_mask'].shape)
                        #if kwargs['past_key_value'] is not None and model.eval():
                            #print(kwargs['past_key_value'][0].shape,kwargs['attention_mask'].shape)
                        kwargs['past_key_value']=None
                        if not self.using_cross: return backup_cross_self_attn_forward_functions[layer_id](*args, **kwargs)
                        if kwargs['past_key_value'] is None:
                            kwargs['past_key_value'] = self.expand_to_batchsize_for_layer(self.past_key_values[2], batch_size, layer_id).to(device)
                            #if model.eval(): print(kwargs['past_key_value'][0].shape)

                        if kwargs['past_key_value'][0].size(-2) + kwargs['key_value_states'].size(-2) == kwargs['attention_mask'].size(-1):
                            pass
                        elif kwargs['past_key_value'][0].size(-2) + kwargs['key_value_states'].size(-2) == kwargs['attention_mask'].size(-1) +self.num_token:
                            # print(kwargs['past_key_value'])
                            am = kwargs['attention_mask']
                            kwargs['attention_mask'] = torch.cat([torch.zeros((*am.shape[:-1],self.num_token), dtype = am.dtype,device=am.device), am], dim=-1)
                        else:
                            raise RuntimeError("Size not match: past length: {}, inputlength:{},\
                                attention mask length {}".format(kwargs['past_key_value'][0].size(-2),
                                kwargs['key_value_states'].size(-2),kwargs['attention_mask'].size(-1)))

                        return backup_cross_self_attn_forward_functions[layer_id](*args, **kwargs)
                    layer_module.layer[1].forward = partial(modified_cross_self_attn_forward, layer_id=i)

        elif isinstance(model, GPT2LMHeadModel):
            pass
        else:
            raise NotImplementedError
        self.plm_modified = True

