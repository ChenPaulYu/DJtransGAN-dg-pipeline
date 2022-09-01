import torch
import numpy as np
import IPython.display as ipd 

from pipeline.config import settings
from pipeline.utils  import squeeze_dim

def wave_visualize(waveform, plt, title=None, timestamps=None, **kargs):
    if isinstance(waveform, torch.Tensor): 
        waveform   = squeeze_dim(waveform).numpy()
    if isinstance(timestamps, torch.Tensor): 
        timestamps = timestamps.numpy()
        
    N = len(waveform)
    T = len(waveform)/settings.SR
    x = np.linspace(0.0, T, N)
    y = waveform
    
    plt.plot(x, y, **kargs)
    if timestamps is not None:
        for timestamp in timestamps:
            plt.axvline(x=timestamp, c='red', ls='dashed', lw=0.5)
    if title: 
        plt.set_title(title)
        
        
def audio_visualize(waveform): 
    if isinstance(waveform, torch.Tensor): waveform=squeeze_dim(waveform).numpy()
    ipd.display(ipd.Audio(waveform, rate=settings.SR))