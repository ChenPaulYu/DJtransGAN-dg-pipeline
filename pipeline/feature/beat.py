import numpy as np
from madmom.features.downbeats import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor
from joblib import Memory

from pipeline.config  import settings
from pipeline.feature import filter_beat

memory = Memory(settings.CACHE_DIR, verbose=1)

@memory.cache
def estimate_beat(audio):
    try:
        proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
        act  = RNNDownBeatProcessor()(audio)
        proc_res  = proc(act) 
        beat     = set(proc_res[:,1])
        downbeat = proc_res[proc_res[:,1]==1,0]

        sig = int(max(beat))
        beat_curve = proc_res[:,0]
        if sig == 6:
            sig_d = 8
        else:
            sig_d = 4   
                    
        return (sig, sig_d), estimate_bpm(beat_curve), filter_beat(beat_curve, downbeat, sig_d), downbeat
    
    except Exception as e:
        print(e)
        return ()
    

@memory.cache
def estimate_bpm(beat_curve):
    try:
        total_beat = len(beat_curve)
        st = int(total_beat/3) 
        ed = int(total_beat*2/3)
        beat_num = ed - st
        total_time = beat_curve[ed] - beat_curve[st] 
        bpm = float(beat_num * 60 / total_time)
        return bpm
    except:
        return -1