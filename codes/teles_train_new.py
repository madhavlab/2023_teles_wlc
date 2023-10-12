#!/usr/bin/env python
# coding: utf-8

# In[1]:
#https://github.com/carrierlxk/py-DSLT/blob/master/Train/SiamNet.py - refer this for shrinkage loss

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from torch.autograd import Variable
from torchmetrics.functional import mean_absolute_error


import pandas as pd
import os
import re
#from tqdm.notebook import tqdm
from tqdm import tqdm
import gc
from numpy.testing import assert_almost_equal
import math
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import yaml
import argparse

from ASR_conformer_model import ConformerEncoder, LSTMDecoder
from conformer_utils import *
from teles_model import *
from teles_utils import *
from teles_metrics import *


# In[2]:

parser = argparse.ArgumentParser(description='TeLeS training')
parser.add_argument('--config', required=True, dest='config_file',
                    help='Requires full path to config file')

args = parser.parse_args()

with open(args.config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    
DEVICE= config['DEVICE']
TRAIN_BS= config['TRAIN_BS']
TEST_BS= config['TEST_BS']
EPOCHS= config['EPOCHS']
asr_checkpoint_path= config['asr_checkpoint_path']
confid_model_name= config['confid_model']
input_dim= config['input_dim']
hidden_dim= config['hidden_dim']
learning_rate= config['learning_rate']
teles_train_dataset= config['teles_train_dataset']
teles_dev_dataset= config['teles_dev_dataset']
best_loss_checkpoint= config['best_loss_checkpoint']
checkpoint= config['checkpoint']
alpha= config['alpha']
beta= config['beta']
loss= config['loss']
SpecAug= config['SpecAug']
a= config['shrinkage_loss_a']
c= config['shrinkage_loss_c']

if SpecAug:
    time_masks = [torchaudio.transforms.TimeMasking(time_mask_param=20, p=0.05) for _ in range(10)]
    validation_transform = nn.Sequential(
     torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80, hop_length=160), #80 filter banks, 25ms window size, 10ms hop
   torchaudio.transforms.FrequencyMasking(freq_mask_param=27),
     *time_masks,)
else:
    validation_transform = transforms.MelSpectrogram(sample_rate=16000, 
                                                 n_mels=80, 
                                                 hop_length=160)
                                                 
validation_transform1 = transforms.MelSpectrogram(sample_rate=16000, 
                                                 n_mels=80, 
                                                 hop_length=160)                                                 

# In[3]:



char_dict = load_char_dict(asr_checkpoint_path, DEVICE)
print(char_dict)


# In[4]:


char_list = list(char_dict.keys())
print(char_list)


# In[5]:


encoder_params = {
    "d_input": 80,
    "d_model": 144,
    "num_layers": 16,
    "conv_kernel_size": 32,
    "dropout": 0.1,
    "num_heads": 4
}

decoder_params = {
    "d_encoder": 144,
    "d_decoder": 320,
    "num_layers": 1,
    "num_classes":len(char_list)+1
}


# In[6]:


encoder = ConformerEncoder(
                      d_input=encoder_params['d_input'],
                      d_model=encoder_params['d_model'],
                      num_layers=encoder_params['num_layers'],
                      conv_kernel_size=encoder_params['conv_kernel_size'], 
                      dropout=encoder_params['dropout'],
                      num_heads=encoder_params['num_heads']
                    )
  
decoder = LSTMDecoder(
                  d_encoder=decoder_params['d_encoder'], 
                  d_decoder=decoder_params['d_decoder'], 
                  num_layers=decoder_params['num_layers'],
                    num_classes= decoder_params['num_classes'])

encoder = encoder.to(DEVICE)
decoder = decoder.to(DEVICE)


# In[7]:


load_model_inference(encoder, decoder, asr_checkpoint_path, DEVICE)


# In[8]:



char_decoder = CTCGreedyCharacterDecoder().to(DEVICE)
#char_decoder = GreedyCharacterDecoder().to(DEVICE)
if confid_model_name=='linear':
    confid_model = teles_linear(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
if confid_model_name=='linear_deep':
    confid_model = teles_linear_new(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
if confid_model_name=='lstm':
    confid_model = teles_lstm(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)


print(confid_model)


# In[9]:


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target, a, c):
        #target = torch.LongTensor(target)
        l2criterion = nn.MSELoss(reduction='mean')
        l2loss = l2criterion(output, target)
        l1criterion = nn.L1Loss(reduction='mean')
        l1loss = l1criterion(output, target)
        shrinkage = (1 + (a*(c-l1loss)).exp()).reciprocal()
        loss = shrinkage * l2loss * output.exp()
        #check if output.exp() is required
        #print(loss)
        loss = torch.mean(loss)
        #print(loss)
        
        return loss


# shrinkage loss = l2/(1 + exp (a · (c − l)))
# reference - https://ieeexplore.ieee.org/document/9273227
# reference - https://github.com/carrierlxk/py-DSLT/blob/master/Train/SiamNet.py

# In[10]:

if loss == 'mae':
    criterion_confid = nn.L1Loss(reduction='mean').to(DEVICE)
if loss == 'shrinkage':
    criterion_confid = CustomLoss().to(DEVICE)

optimizer_confid = torch.optim.Adam(confid_model.parameters(), lr=learning_rate, 
                       weight_decay=1e-05, betas=(0.9, 0.98), eps=1e-9)


# In[11]:


teles_train_df = pd.read_csv(teles_train_dataset, sep='\t', header=None)
teles_dev_df = pd.read_csv(teles_dev_dataset, sep='\t', header=None)

teles_train_df = teles_train_df.sample(frac=1).reset_index(drop=True)
teles_dev_df = teles_dev_df.sample(frac=1).reset_index(drop=True)

teles_trainset = TelesDataset(teles_train_df, char_dict, transform = validation_transform)

teles_trainloader = torch.utils.data.DataLoader(teles_trainset, batch_size = TRAIN_BS, shuffle = True, 
                                          collate_fn = collate_batch_teles, drop_last=True, 
                                          num_workers=4, pin_memory=True)

teles_devset = TelesDataset(teles_dev_df, char_dict, transform = validation_transform1)

teles_devloader = torch.utils.data.DataLoader(teles_devset, batch_size = TEST_BS, shuffle = True, 
                                          collate_fn = collate_batch_teles, drop_last=True, 
                                          num_workers=4, pin_memory=True)


# In[12]:



def actual_score_gen(references, sts, ets, actual_predictions, uncollapsed_predictions, alpha, beta):
    
    actual_score_list = []
    new_ops_list, ref_list, hyp_list = align_gen(references, actual_predictions)
    for z in range(len(actual_predictions)):
    
        pred = uncollapsed_predictions[z]
        coll_pred = actual_predictions[z]
        opcodes = new_ops_list[z]
        ref = ref_list[z]
        hyp = hyp_list[z]
        st = sts[z]
        et = ets[z]
        
        pred_red = re.sub(' +',' ',pred)
        pred_split = pred_red.split(' ')

        coll_pred_red = re.sub(' +',' ',coll_pred)
        coll_pred_split = coll_pred_red.split(' ')


        pred_split_final = []

        for k in range(len(pred_split)):
            length = len(pred_split[k])
            flag = 0
            for charac in pred_split[k]:
                if charac == '*':
                    flag +=1
            if flag == length:
                continue
            else:
                pred_split_final.append(pred_split[k])

        pred_ctm = []   
        end_frame = 0
        for i in range(len(pred_split_final)):
            confid_input = []
            start_frame = end_frame + pred[end_frame:].find(pred_split_final[i])
            #start_frame = pred.find(pred_split_final[i])
            end_frame = start_frame + len(pred_split_final[i]) - 1
            start_time = start_frame*4/100
            end_time = end_frame*4/100
            pred_ctm.append([coll_pred_split[i], start_time, end_time])


        actual_score = []

        conf = 0
 
        gt_index = 0
        pred_index = 0
        for i in range(len(opcodes)):
            if(opcodes[i]=='C'):
                similarity_index = 1
        
                overlap = get_alignment_score([ref[gt_index], st[gt_index], et[gt_index]], pred_ctm[pred_index])
                gt_index += 1
                pred_index += 1
                conf = alpha*similarity_index + ((1-alpha)*overlap)
                actual_score.append(conf)
        
        
            elif(opcodes[i]=='I'):
                conf = 0
                actual_score.append(conf)
        
        
            elif(opcodes[i]=='D'):      
                conf = 0
                gt_index +=1
        
        
            elif(opcodes[i]=='S'):
                similarity_index = get_jaccard(ref[i], hyp[i])
                overlap = get_alignment_score([ref[gt_index], st[gt_index], et[gt_index]], pred_ctm[pred_index])
                gt_index += 1
                pred_index += 1 
                conf = beta*similarity_index + ((1-beta)*overlap)
                actual_score.append(conf)
        
        actual_score_list.append(actual_score)
        
    return actual_score_list, new_ops_list, ref_list, hyp_list


# In[13]:


def train(confid_model, optimizer_confid, criterion_confid, batch, encoder, decoder, char_decoder, char_list, alpha, beta, device):
    confid_model.train()
    optimizer_confid.zero_grad()
    
    input_tens, input_len_list, references, sts, ets, mask = batch
        
    attention_cem, decoder_cem, soft_cem, final_predictions, uncollapsed_predictions = ConformerForward(encoder, decoder, char_decoder, input_tens, input_len_list, mask, char_list, device= DEVICE)
    
    #Generate actual_score
    
    try:
    
        actual_score_list, opcodes, _ ,_  = actual_score_gen(references, sts, ets, final_predictions, uncollapsed_predictions, alpha, beta)
    except:
        print("exception1")
        loss_confid = torch.zeros(1,1)

        return loss_confid.item()
    
    pred_split_final_list = []
    
    for i in range(len(uncollapsed_predictions)):  
        pred_split_final = []
        pred = uncollapsed_predictions[i]
        pred_red = re.sub(' +',' ',pred)
        pred_split = pred_red.split(' ')

        for k in range(len(pred_split)):
            length = len(pred_split[k])
            flag = 0
            for charac in pred_split[k]:
                if charac == '*':
                    flag +=1
            if flag == length:
                continue
            else:
                pred_split_final.append(pred_split[k])
        pred_split_final_list.append(pred_split_final)
    
    confid_tensor = []
    final_loss = 0
    score_seq_tens = []
    input_stack = []
    
    for j in range(len(uncollapsed_predictions)):
        pred = uncollapsed_predictions[j]
        end_frame=0
        for i in range(len(pred_split_final_list[j])):
            confid_input = []
            start_frame = end_frame + pred[end_frame:].find(pred_split_final_list[j][i])
            end_frame = start_frame + len(pred_split_final_list[j][i]) - 1
            indices = [*range(start_frame, end_frame+1, 1)]
            attention_input = attention_cem[j][indices]
            decoder_input = decoder_cem[j][indices]
            soft_input = soft_cem[j][indices]
            attention_input = torch.mean(attention_input, 0, False) #If false, shape is [256]. If True, shape is [1,256]
            decoder_input = torch.mean(decoder_input, 0, False)
            soft_input = torch.mean(soft_input, 0, False)
            confid_input = [attention_input, decoder_input, soft_input]
            confid_tensor = torch.cat(confid_input, dim=-1)
            input_stack.append(confid_tensor)
            
    try:
        input_stack = torch.stack(input_stack)


        score = confid_model(input_stack)
        actual_score = []

        for sublist in actual_score_list:
            for item in sublist:
                actual_score.append(item)

        target_score = Variable(torch.FloatTensor(actual_score), requires_grad = True).to(device)

        if loss == 'mae':
            loss_confid = criterion_confid(score.squeeze(), target_score)
        if loss == 'shrinkage':
            loss_confid = criterion_confid(score.squeeze(), target_score, a, c)
        #loss_confid = criterion_confid(score.squeeze(), target_score)

        #loss_confid = criterion_confid(score, target_score)

        loss_confid.backward(retain_graph=True)


        torch.nn.utils.clip_grad_norm_(confid_model.parameters(), 0.5)
        optimizer_confid.step()
    except:
        loss_confid = torch.zeros(1,1)
#     final_loss = final_loss + loss_confid.item()
        
#     return final_loss/len(pred_split)
    return loss_confid.item()


# In[14]:


def test_validate(confid_model, criterion_confid, batch, encoder, decoder, char_decoder, char_list, epoch_num, alpha, beta, device):
    confid_model.eval()
    
    input_tens, input_len_list, references, sts, ets, mask = batch
        
    attention_cem, decoder_cem, soft_cem, final_predictions, uncollapsed_predictions = ConformerForward(encoder, decoder, char_decoder, input_tens, input_len_list, mask, char_list, device= DEVICE)
    
    actual_score_list, new_ops_list, _ ,_ = actual_score_gen(references, sts, ets, final_predictions, uncollapsed_predictions, alpha, beta)

    pred_split_final_list = []
    
    for i in range(len(uncollapsed_predictions)):  
        pred_split_final = []
        pred = uncollapsed_predictions[i]
        pred_red = re.sub(' +',' ',pred)
        pred_split = pred_red.split(' ')

        for k in range(len(pred_split)):
            length = len(pred_split[k])
            flag = 0
            for charac in pred_split[k]:
                if charac == '*':
                    flag +=1
            if flag == length:
                continue
            else:
                pred_split_final.append(pred_split[k])
        pred_split_final_list.append(pred_split_final)

    confid_tensor = []
    final_loss = 0
    score_seq_tens = []
    input_stack = []
    
    final_jsd = 0
        
    for j in range(len(uncollapsed_predictions)):
        pred = uncollapsed_predictions[j]
        end_frame = 0
        for i in range(len(pred_split_final_list[j])):
            confid_input = []
            start_frame = end_frame + pred[end_frame:].find(pred_split_final_list[j][i])
            end_frame = start_frame + len(pred_split_final_list[j][i]) - 1
            indices = [*range(start_frame, end_frame+1, 1)]
            attention_input = attention_cem[j][indices]
            decoder_input = decoder_cem[j][indices]
            soft_input = soft_cem[j][indices]
            attention_input = torch.mean(attention_input, 0, False) #If false, shape is [256]. If True, shape is [1,256]
            decoder_input = torch.mean(decoder_input, 0, False)
            soft_input = torch.mean(soft_input, 0, False)
            confid_input = [attention_input, decoder_input, soft_input]
            confid_tensor = torch.cat(confid_input, dim=-1)

            input_stack.append(confid_tensor)
    
    with torch.no_grad():
        input_stack = torch.stack(input_stack)

        score = confid_model(input_stack)#1.unsqueeze(1))   #Tensor

        actual_score = [] # List

        for sublist in actual_score_list:
            for item in sublist:
                actual_score.append(item)

        target_score = Variable(torch.FloatTensor(actual_score), requires_grad = True).to(device)
        if loss == 'mae':
            loss_confid = criterion_confid(score.squeeze(), target_score)
        if loss == 'shrinkage':
            loss_confid = criterion_confid(score.squeeze(), target_score, a, c)
    

    #print(f"L1 Loss : {loss_confid}")

#     final_loss = final_loss + loss_confid.item()
    if(epoch_num%2 == 0):
#         counter = 0
#         for z in range(5):
        print("ref:", references)
        print("hyp:", final_predictions)
        print("Act. score:", actual_score)
        print("Pred score:", score.tolist())
#             score_temp = score.tolist()
#             print("Pred. score:", score_temp[counter:counter+len(final_predictions[z])-1])
#             counter = counter+len(final_predictions[z])
                  
        

#     return final_loss/len(pred_split)
    return loss_confid.item(), score, target_score, new_ops_list


# In[15]:





# In[16]:


best_loss = float('inf')
for i in tqdm(range(EPOCHS)):
    exception_counter = 0
    print(f"Epoch : {i+1}/{EPOCHS}")
    train_loss = 0
    num_train_batches = 0
    final_cer = []
    final_score = []
    
    for batch in tqdm(teles_trainloader):
        
        final_loss = train(confid_model, optimizer_confid, criterion_confid,
                           batch, encoder, decoder, char_decoder, char_list, alpha, beta, DEVICE)
      #  except:
      #      exception_counter += 1
      #      continue
        train_loss = train_loss + final_loss
        num_train_batches += 1
        
    final_train_loss = train_loss/num_train_batches

    val_loss = 0
    num_test_batches = 0
    
    
    
    for batch in tqdm(teles_devloader):
     #   try:
        final_loss, score, target_score, opcodes = test_validate(confid_model, criterion_confid,
                                   batch, encoder, decoder, char_decoder, char_list, i, alpha, beta, DEVICE)
     #   except:
     #       exception_counter += 1
     #       continue
            
        val_loss = val_loss + final_loss
        num_test_batches += 1
        
    final_val_loss = val_loss/num_test_batches
    
    if final_val_loss <= best_loss:
        print('Validation loss improved, saving checkpoint.')
        best_loss = final_val_loss
        save_checkpoint_confid(confid_model, optimizer_confid, final_val_loss, i+1, 
                               best_loss_checkpoint)
    save_checkpoint_confid(confid_model, optimizer_confid, final_val_loss, i+1, 
                               checkpoint)
    print(f"Train loss : {final_train_loss}")
    print(f"Val loss : {final_val_loss}")
    print(f"Exceptions : {exception_counter}")

# In[ ]:




