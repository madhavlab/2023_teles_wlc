import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset
from torchmetrics.text.wer import WordErrorRate
from torchmetrics.text.cer import CharErrorRate
from torch.autograd import Variable

import pandas as pd
import os
import re
import gc
from numpy.testing import assert_almost_equal
import math
import numpy as np
import jiwer

from conformer_utils import int_to_text_uncollapse, int_to_text_collapse, int_to_text_final

def ConformerForward(encoder, decoder, char_decoder, spectrograms, input_len_list, mask, char_list, device='cuda:0'):
    ''' Evaluate model on test dataset. '''

    encoder.eval()
    decoder.eval()

    # Move to GPU
    spectrograms = spectrograms.to(device)

    mask = mask.to(device)

    with torch.no_grad():
        outputs, attention_cem = encoder(spectrograms, mask)
        outputs, decoder_cem = decoder(outputs)
 
        soft_cem = F.softmax(outputs, dim=-1)


        inds, uncollapsed_inds = char_decoder(outputs.detach(), input_len_list)
        
        uncollapsed_predictions = []
        
        for sample1 in uncollapsed_inds:
            uncollapsed_predictions.append(int_to_text_uncollapse(sample1, len(char_list), char_list))

        collapse_predictions = []
        final_predictions = []
        for sample in inds:
            collapse_predictions.append(int_to_text_collapse(sample, len(char_list), char_list))
            final_predictions.append(int_to_text_final(sample, len(char_list), char_list))

    return attention_cem, decoder_cem, soft_cem, final_predictions, uncollapsed_predictions
    
class TelesDataset(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """
    def __init__(self, data_frame, char_dict, transform=None):
        self.data_frame = data_frame
        #self.root_dir = root_dir
        self.transform = transform
        self.char_dict = char_dict
        #self.max_transcript_len = N
    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_path = self.data_frame.iloc[idx, 0]
        waveform, sample_rate = torchaudio.load(file_path, normalize = True)
        channel_dim = waveform.shape[0]
        if channel_dim > 1:
            waveform = torch.mean(waveform, 0, keepdim=True)
        if sample_rate!=16000:
            waveform = transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        mel_spec = self.transform(waveform) # (channels, n_mels, time)
        mel_spec = mel_spec.squeeze(0).transpose(0,1)  # (time, n_mels)
        mel_spec_len = ((mel_spec.shape[0] - 1) // 2 - 1) // 2
        
        transcript = self.data_frame.iloc[idx,1]
        transcript = str(transcript)
        
        st = self.data_frame.iloc[idx,2].split()
        et = self.data_frame.iloc[idx,3].split()
        
        st = [float(st[i]) for i in range(len(st))]
        et = [float(et[i]) for i in range(len(et))]
        
        return mel_spec, mel_spec_len, transcript, st, et
    
def collate_batch_teles(batch):
    
    input_list, input_len_list, references, sts, ets = [], [], [], [], []

    for (_input, _input_len,  _transcript, _st, _et) in batch:
        input_list.append(_input)
        input_len_list.append(_input_len)
        references.append(_transcript)
        sts.append(_st)
        ets.append(_et)
        #print(_input.shape, _label.shape, _input_len, _label_len)
    input_tens = nn.utils.rnn.pad_sequence(input_list, batch_first=True)    
    
    mask = torch.ones(input_tens.shape[0], input_tens.shape[1], input_tens.shape[1])
    for i, l in enumerate(input_len_list):
        mask[i, :, :l] = 0
    return input_tens, input_len_list, references, sts, ets, mask.bool()

def get_alignment_score(ref, hyp):
    score = 1-((abs(ref[1]-hyp[1])+abs(ref[2]-hyp[2]))/(ref[2]-ref[1]))
    
    score = max(0,score)
    if score <0 or score >1:
        print('\n'.join([str(ref), str(hyp), str(score)]))
    return score

def get_jaccard(ref, hyp):
    list1 = list(ref)
    list2 = list(hyp)
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def align_gen(references, final_predictions):
    new_ops_list = []
    ref_list = []
    hyp_list = []
    for i in range(len(final_predictions)):
        alignments = jiwer.visualize_alignment(jiwer.process_words(references[i], final_predictions[i]), show_measures=False, skip_correct=False) #jiwer 3.0.2 had issue, so added skip_correct
        alignments = alignments.split("\n")

        ref = alignments[1][5:]
        hyp = alignments[2][5:]
        
        ref = re.sub(" +", " ", ref)
        hyp = re.sub(" +", " ", hyp)

        ref = ref.split()
        hyp = hyp.split()

        new_ops = []
        for i in range(len(ref)):
            if hyp[i]==ref[i]:
                new_ops.append('C')
            elif '*' in hyp[i]:
                new_ops.append("D")
            elif '*' in ref[i]:
                new_ops.append("I")
            else:
                new_ops.append("S")
        new_ops_list.append(new_ops)
        ref_list.append(ref)
        hyp_list.append(hyp)
        
    return new_ops_list, ref_list, hyp_list

def save_checkpoint_confid(model, optimizer, valid_loss, epoch, checkpoint_path):
    ''' Save model checkpoint '''
    torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'encoder_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path)
