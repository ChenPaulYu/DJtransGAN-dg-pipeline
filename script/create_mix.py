import warnings
warnings.filterwarnings('ignore')

import os
import sys
if os.getcwd() in sys.path:
    sys.path.append('../')
else:    
    sys.path.append(os.getcwd())    

import torch
import argparse
from tqdm             import tqdm
from joblib           import Parallel, delayed
from pipeline.config  import settings
from pipeline.utils   import load_audio, load_json, out_npy, find_nearest, find_index, check_exist
from pipeline.feature import estimate_feature
from pipeline.segment import get_segment

def get_cue_points(downbeat, cue):    
    cue_out_idx = find_nearest(downbeat, cue)
    return [downbeat[cue_out_idx-settings.CUE_BAR], downbeat[cue_out_idx]]

def create_mix_obj(mix_id, cue, audio_dir, out_dir):
    
    audio_path = os.path.join(audio_dir, f'{mix_id}.wav')
    audio      = load_audio(audio_path)
    
    feature    = estimate_feature(audio_path)
    sig, bpm, beat, downbeat = feature['beat']
    cue_points = get_cue_points(downbeat, cue)
    
    mix_obj    = {
        **feature, 
        'id'   : mix_id,
        'cue'  : cue_points, 
    }
    
    out_path = os.path.join(out_dir, f'{mix_id}.npy')
    check_exist(out_path)
    out_npy(mix_obj, out_path)
    
    
    
def main():
    parser    = argparse.ArgumentParser(description='Create Mix Object')
    parser.add_argument('--n_core' , help='core number of mutliprocessor', type=int , default=5)
    
    args      = parser.parse_args()
    out_dir   = os.path.join(settings.MIX_DIR, 'obj')
    audio_dir = os.path.join(settings.MIX_DIR, 'audio')
    meta_json = load_json(os.path.join(settings.MIX_DIR, 'meta.json'))
    
    print('Create mix begin ...')
    Parallel(n_jobs=args.n_core)(delayed(create_mix_obj)(mix_id, 
                                                         meta_json[mix_id]['cue'], 
                                                         audio_dir, out_dir) for mix_id in tqdm(meta_json)) 
    print('Create mix complete ...')
    


if __name__ == "__main__":
    main()
    