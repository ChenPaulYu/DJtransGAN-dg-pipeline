import os
import torch

from pipeline.config     import settings
from pipeline.utils      import load_npy, load_audio
from pipeline.utils      import get_stretch_ratio, get_key_diff, time_stretch, pitch_shift
from pipeline.mixability import split_audio


def sync_cue(src_audio, tgt_audio, src_cue, tgt_cue):   
    
    # time inverse inversely proportional with bpm
    ratio         = get_stretch_ratio(src_cue[1]-src_cue[0], tgt_cue[1]-tgt_cue[0])
    splited       = split_audio(tgt_audio, tgt_cue)
    splited[1]    = time_stretch(splited[1], ratio)
    begin_idx     = 1 if splited[0] is None else 0
    synced        = splited[begin_idx]
    tgt_cue       = [tgt_cue[0], 
                     tgt_cue[0]+(src_cue[1]-src_cue[0])]
    
    for s in splited[begin_idx+1:]:
        synced    = torch.cat((synced, s), 1)
        
    return  synced, tgt_cue

def sync_pair(src_obj, tgt_obj, src_audio, tgt_audio):
    
    pair_obj   = {}
    pair_audio = {}

    ratio      = get_stretch_ratio(tgt_obj['bpm'], src_obj['bpm'])
    diff       = get_key_diff(tgt_obj['key'], src_obj['key'])
    
    pair_obj['prev'] = {
        **src_obj, 
        'cue': src_obj['cue'][1], 
        'source': src_obj['source'][1],
    }
    

    pair_obj['next'] = {
        **tgt_obj, 

        'key'     : src_obj['key'],
        'bpm'     : src_obj['bpm'],
        'beat'    : tgt_obj['beat']     / ratio,
        'downbeat': tgt_obj['downbeat'] / ratio, 
        'source'  : tgt_obj['source'][0],
        'cue'     : [c/ratio for c in tgt_obj['cue'][0]]
    }
    
    pair_audio['prev'] = src_audio
    pair_audio['next'], pair_obj['next']['cue'] = sync_cue(src_audio, 
                                                           time_stretch(pitch_shift(tgt_audio, diff), ratio),
                                                           pair_obj['prev']['cue'],
                                                           pair_obj['next']['cue'])
    
    
    return pair_obj, pair_audio