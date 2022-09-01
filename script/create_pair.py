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
from tqdm                import tqdm
from joblib              import Parallel, delayed

from pipeline.config     import settings
from pipeline.utils      import load_audio, load_npy, out_audio, out_json
from pipeline.utils      import check_exist, random_samples, time_to_str, time_to_samples
from pipeline.utils      import wave_visualize, audio_visualize
from pipeline.mixability import get_seg_ids, get_cand_ids
from pipeline.mixability import match_pair , sync_pair
    
    
    
def obj_to_meta(pair_obj):
    meta = {
        'id'  : pair_obj['pair_id'],
        'prev': {
            'cue'   : pair_obj['prev']['cue'],
            'source': pair_obj['prev']['source']
        }, 
        'next': {
            'cue'   : pair_obj['next']['cue'],
            'source': pair_obj['next']['source']
        }
    }
    return meta
    

    
def create_pair(src_id, seg_ids, n_sample, match_type, out_dir, seg_dir):
    
    obj_dir   = os.path.join(seg_dir, 'obj')
    audio_dir = os.path.join(seg_dir, 'audio')
    cand_ids  = get_cand_ids(src_id, seg_ids)
    matched   = match_pair(src_id, cand_ids, n_sample=n_sample, match_type=match_type) 
        
    if matched is None:
        return 
    else:
        src_id, tgt_id = matched

    src_obj   = load_npy(os.path.join(obj_dir, f'{src_id}.npy')).item()
    src_audio = load_audio(os.path.join(audio_dir, f'{src_id}.wav'))

    tgt_obj   = load_npy(os.path.join(obj_dir, f'{tgt_id}.npy')).item()
    tgt_audio = load_audio(os.path.join(audio_dir, f'{tgt_id}.wav'))
        
    pair_obj, pair_audio = sync_pair(src_obj, tgt_obj, src_audio, tgt_audio)
    pair_id   = f'{src_id}__{tgt_id}'
    pair_obj['pair_id'] = pair_id
        
    for key in pair_audio:
        out_path = os.path.join(out_dir, 'audio', pair_id, f'{key}.wav')
        check_exist(out_path)
        out_audio(pair_audio[key].clone().to(torch.float32), out_path)
    
    out_path = os.path.join(out_dir, 'meta', f'{pair_id}.json')
    check_exist(out_path)
    out_json(obj_to_meta(pair_obj), out_path)
        
    
    
def main():
    parser    = argparse.ArgumentParser(description='Create Mixabile Pair')
    parser.add_argument('--match'     , help='choose match type e.g: all, nn, rule', type=str , default=None)
    parser.add_argument('--n_sample'  , help='sample number of NN to speed up', type=int , default=10)
    parser.add_argument('--n_core'    , help='core number of mutliprocessor'  , type=int , default=5)
    
    args      = parser.parse_args()
    seg_ids   = get_seg_ids() 
    
    seg_dir   = os.path.join(settings.TRACK_DIR, 'seg')
    out_dir   = os.path.join(settings.TRACK_DIR, 'pair')

    print('Pair create begin ...')
    if args.match in ['all', 'nn']:
        for src_id in tqdm(seg_ids):
            create_pair(src_id, seg_ids, args.n_sample, args.match, out_dir)
    else:
        Parallel(n_jobs=args.n_core)(delayed(create_pair)(src_id, 
                                                          seg_ids, 
                                                          args.n_sample, 
                                                          args.match, 
                                                          out_dir, 
                                                          seg_dir) for src_id in tqdm(seg_ids))  
    
    print('Pair create complete ...')
    



if __name__ == "__main__":
    main()
    