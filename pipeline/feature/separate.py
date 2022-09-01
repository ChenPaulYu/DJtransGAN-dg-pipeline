import os
import torch
import librosa
import openunmix
import numpy as np
import torch.nn as nn
from joblib         import Memory

from pipeline.config  import settings
from pipeline.utils   import load_json, out_json, load_audio, load_npy, out_npy
from pipeline.utils   import get_extention, to_mono, get_device, check_exist, squeeze_dim, time_to_samples

SEP_SOURCES = ['vocals', 'drums', 'bass', 'other']
STEM_DIR    = os.path.join(settings.CACHE_DIR, 'stem')

class Separator(nn.Module):
    def __init__(self, model='umxl', n_gpu=-1, mono=True):
        super().__init__()
        self.mono   = mono
        self.device = get_device(n_gpu)
        self.model  = openunmix.utils.load_separator(model, device=self.device, pretrained=True)
        self.model.sample_rate = torch.tensor(settings.SR)
        self.model.freeze()
        
        
    def forward(self, audio):
        audio  = openunmix.utils.preprocess(audio)
        audio  = self.model(audio.to(self.device))
        return to_mono(audio).cpu() if self.mono else audio
    
    
def get_separator(n_gpu=-1):
    return Separator(n_gpu=n_gpu)
    

def estimate_stem(audio_path, separator):
    tmp_path  = os.path.join(STEM_DIR, '.stem_cache.json')
    check_exist(tmp_path)
    tmp_json  = load_json(tmp_path) if os.path.exists(tmp_path) else {}
    audio_id  = os.path.splitext(audio_path)[0].split('/')[-1]  
    if audio_id in tmp_json.keys() and os.path.exists(tmp_json[audio_id]):
        stem  = load_npy(tmp_json[audio_id])
    else:
        audio = load_audio(audio_path)
        stem  = separator(audio).squeeze(0)
        tmp_json[audio_id] = os.path.join(STEM_DIR, f'{audio_id}.npy')
        check_exist(tmp_json[audio_id])
        out_json(tmp_json, tmp_path)
        out_npy(stem, tmp_json[audio_id])
        
    return stem

def estimate_sources(stem):
    estimated = []
    for idx, source in enumerate(SEP_SOURCES):
        detected = int(estimate_rms(stem[idx]) >= settings.RMS_THRESHOLD)
        if detected:
            estimated.append(source)
    return estimated


def estimate_rms(audio):
    if isinstance(audio, torch.Tensor): 
        audio = squeeze_dim(audio).numpy()
    rms         = librosa.feature.rms(y=audio)
    return np.mean(rms)
    