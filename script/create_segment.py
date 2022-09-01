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
from joblib           import Parallel, delayed, Memory

from pipeline.config  import settings
from pipeline.utils   import check_exist, check_extent, out_json, out_npy, out_audio
from pipeline.feature import estimate_feature, estimate_stem, get_separator
from pipeline.segment import get_segment


def create_segment(audio_path, seg_dir, separator):
    segs    = get_segment(audio_path, separator)    
    for seg in segs:
        if seg['info']['key'] != 1:
            if seg['info']['sig'][0] == 4 and seg['info']['sig'][1] == 4:
                if len(seg['obj']['downbeat']) > 8:
                
                    seg_id     = seg['seg_id']
                    obj_path   = os.path.join(seg_dir, 'obj'  , f'{seg_id}.npy')
                    info_path  = os.path.join(seg_dir, 'meta' , f'{seg_id}.json')
                    audio_path = os.path.join(seg_dir, 'audio', f'{seg_id}.wav')

                    check_exist(obj_path)
                    check_exist(info_path)
                    check_exist(audio_path)

                    out_npy(seg['obj']    , obj_path)
                    out_json(seg['info']  , info_path)
                    out_audio(seg['audio'], audio_path)
    

    
def main():
    parser    = argparse.ArgumentParser(description='Create Segment from Musical Structure')
    parser.add_argument('--feature', help='whether do feature extraction'   , type=int , default=0)
    parser.add_argument('--stem'   , help='whether do stem decomposition'   , type=int , default=0)
    parser.add_argument('--segment', help='whether do segment creation'     , type=int , default=1)
    parser.add_argument('--n_core' , help='core number of mutliprocessor'   , type=int , default=5)
    parser.add_argument('--n_gpu'  , help='gpu number of stem decomposition', type=int , default=-1)
    
    args        = parser.parse_args()
    seg_dir     = os.path.join(settings.TRACK_DIR, 'seg')
    audio_dir   = os.path.join(settings.TRACK_DIR, 'audio')
    audio_paths = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if check_extent(file, ['wav', 'mp3'])]
    separator   = get_separator(n_gpu=args.n_gpu)
    
    if args.feature:
        print('Feature extract begin ...')
        Parallel(n_jobs=args.n_core)(delayed(estimate_feature)(audio_path) for audio_path in tqdm(audio_paths))   
        print('Feature extract complete ...')
        
    if args.stem:
        print('Stem decompose begin ...')
        if args.n_gpu == -1:
            Parallel(n_jobs=args.n_core)(delayed(estimate_stem)(audio_path, separator) for audio_path in tqdm(audio_paths))   
        else:
            for audio_path in tqdm(audio_paths):
                estimate_stem(audio_path, separator)
        print('Stem decompose complete ...')
    
    if args.segment:
        print('Create sgment begin ...')
        Parallel(n_jobs=args.n_core)(delayed(create_segment)(audio_path, seg_dir, separator) for audio_path in tqdm(audio_paths)) 
        print('Create sgment complete ...')


if __name__ == "__main__":
    main()
    