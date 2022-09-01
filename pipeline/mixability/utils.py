import os
import torch
import librosa
import numpy as np

from pipeline.config import settings
from pipeline.utils  import squeeze_dim, time_to_samples

SEG_DIR   = os.path.join(settings.TRACK_DIR, 'seg')
OBJ_DIR   = os.path.join(SEG_DIR, 'obj')
META_DIR  = os.path.join(SEG_DIR, 'meta')


def get_seg_ids():
    obj_ids   = [obj.split('.npy')[0]   for obj  in os.listdir(OBJ_DIR)  if obj.endswith('.npy')]
    meta_ids  = [meta.split('.json')[0] for meta in os.listdir(META_DIR) if meta.endswith('.json')]
    return list(set(obj_ids) & set(meta_ids))

def get_cand_ids(src_id, seg_ids):
    track_id = src_id.split('_')[0]
    return [seg_id for seg_id in seg_ids if seg_id.split('_')[0] != track_id]

def split_audio(audio, cue):
    splited      = ()
    cue_sample   = [time_to_samples(c) for c in cue]
    before       = None if cue_sample[0] == 0 else  audio[:, :cue_sample[0]]
    middle       = audio[:, cue_sample[0]:cue_sample[1]]
    after        = None if cue_sample[1] == 0 else audio[:, cue_sample[1]:]
    return [before, middle, after]
    

def read_audio(audio, sr=settings.SR): 
    
    if isinstance(audio, torch.Tensor): 
        audio   = squeeze_dim(audio).numpy()
    
    avgv = np.load(os.path.join(settings.MIXABLE_DIR, 'model', 'avg.npy'))
    stdv = np.load(os.path.join(settings.MIXABLE_DIR, 'model', 'std.npy'))
    S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S = np.transpose(np.log(1+10000*S))
    S = (S-avgv)/stdv
    S = np.expand_dims(S, 2)
    return S