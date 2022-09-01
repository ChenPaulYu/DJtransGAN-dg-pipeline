import torch
import librosa
import numpy as np
import pyloudnorm   as pyln
import pyrubberband as pyrb


from pipeline.config import settings
from pipeline.utils  import squeeze_dim


def get_stretch_ratio(src, tgt):
    return tgt / src

def get_key_diff(src, tgt):
    diff_forward  = tgt - src
    diff_backward = diff_forward - 13 if diff_forward > 0 else 13 + diff_forward
    return diff_forward if abs(diff_forward) < abs(diff_backward) else diff_backward

def time_stretch(audio, ratio, sr=settings.SR):
    if ratio == 1:
        return audio
    if isinstance(audio, torch.Tensor): 
        stretched = torch.from_numpy(pyrb.time_stretch(squeeze_dim(audio).numpy(), sr, ratio)).unsqueeze(0)
    else:
        stretched = pyrb.time_stretch(audio, sr, ratio)
    return stretched

def pitch_shift(audio, step, sr=settings.SR):
    if step == 0:
        return audio
    if isinstance(audio, torch.Tensor): 
        shifted = torch.from_numpy(pyrb.pitch_shift(squeeze_dim(audio).numpy(), sr, step)).unsqueeze(0)
    else:
        shifted = pyrb.pitch_shift(audio, sr, step)
    return shifted