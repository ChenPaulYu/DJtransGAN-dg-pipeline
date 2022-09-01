import re
import os
import torch
import json
import random
import joblib
import librosa
import torchaudio
import numpy as np

from pipeline.config  import settings

random.seed(settings.RANDOM_SEED)

# Transform

def to_mono(audio, dim=-2): 
    if len(audio.size()) > 1:
        return torch.mean(audio, dim=dim, keepdim=True)
    else:
        return audio
    
def time_to_samples(time): 
    return librosa.time_to_samples(time, sr=settings.SR)

def samples_to_time(samples):
    return librosa.samples_to_time(samples, sr=settings.SR)
    

# I/O

def load_audio(audio_path, 
               sr=settings.SR, 
               mono=True):
    if 'mp3' in audio_path:
        torchaudio.set_audio_backend('sox_io')
    audio, org_sr = torchaudio.load(audio_path)
    audio = to_mono(audio) if mono else audio
    
    if org_sr != sr:
        audio = torchaudio.transforms.Resample(org_sr, sr)(audio)

    return audio

def out_audio(data, out_path, sr=settings.SR):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    if len(data.size()) == 1: data = data.unsqueeze(0)
    torchaudio.save(out_path, data, sr)

def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def out_json(data, out_path, cover=True):
    if check_exist(out_path) and cover == False: return
    with open(out_path, 'w') as outfile: json.dump(data, outfile)
        
def out_npy(data, out_path):
    check_exist(out_path)
    np.save(out_path.split('.npy')[0], data)
    
def load_npy(in_path):
    return np.load(in_path, allow_pickle=True)
    
    
# Others

def time_to_str(secs):
    return f'{int(secs/60)}:{(secs%60):.2f}'
    
def str_to_time(string):
    mins, secs = string.split(':')
    return mins*60+secs
    
def check_exist(out_path):    
    if re.compile(r'^.*\.[^\\]+$').search(out_path):
        out_path = os.path.split(out_path)[0]        
    existed = os.path.exists(out_path)
    if not existed:
        os.makedirs(out_path, exist_ok=True)
    return existed

def check_extent(file, exts):
    if isinstance(exts, str):
        return file.endswith(f'.{exts}')
    else:
        return bool(sum([file.endswith(f'.{ext}') for ext in exts]) > 0)

def pair_wise(arr):
    return [(a, b) for (a, b) in zip(arr[:-1], arr[1:])]

def find_nearest(arr, val):
    return np.argmin(abs(arr - val))

def find_index(arr, val):
    return np.where(arr == val)[0][0]

def get_extention(data_path):
    return os.path.splitext(audio_path)[-1][1:]

def random_samples(arr, n_sample):
    return arr if len(arr) < n_sample else random.sample(arr, n_sample)
    
def squeeze_dim(data):
    dims = [i for i in range(len(data.size())) if data.size(i) == 1]
    for dim in dims:
        data = data.squeeze(dim)
    return data

def get_device(n_gpu):
    return torch.device('cpu' if int(n_gpu) == -1 else f'cuda:{n_gpu}')