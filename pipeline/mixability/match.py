import os
import random
from tqdm import tqdm

from pipeline.config     import settings
from pipeline.utils      import load_audio, load_npy, load_json
from pipeline.utils      import random_samples, time_to_samples
from pipeline.mixability import estimiate_compatibility_by_rule, estimate_mixability_by_nn

random.seed(0)

SEG_DIR   = os.path.join(settings.TRACK_DIR, 'seg')
OBJ_DIR   = os.path.join(SEG_DIR, 'obj')
META_DIR  = os.path.join(SEG_DIR, 'meta')
AUDIO_DIR = os.path.join(SEG_DIR, 'audio')

def match_pair_by_rule(src_id, cand_ids):
    matched  = []
    src_meta = load_json(os.path.join(META_DIR, f'{src_id}.json'))
    for cand_id in cand_ids:
        cand_meta     = load_json(os.path.join(META_DIR, f'{cand_id}.json'))
        compatibility = estimiate_compatibility_by_rule(src_meta, cand_meta)
        if compatibility:
            matched.append(cand_id)
    return matched

def match_pair_by_nn(src_id, cand_ids, n_sample=None):
    matched   = []
    cand_objs = []
    
    best      = {'mixability': 0, 'id': None, 'audio': None, 'obj': None}
    if len(cand_ids) == 0:
        return None
    
    src_obj    = load_npy(os.path.join(OBJ_DIR, f'{src_id}.npy')).item()
    src_audio  = load_audio(os.path.join(AUDIO_DIR, f'{src_id}.wav'))

    for cand_id in tqdm(cand_ids if n_sample is None else random_samples(cand_ids, n_sample)):
        cand_obj   = load_npy(os.path.join(OBJ_DIR, f'{cand_id}.npy')).item()
        cand_audio = load_audio(os.path.join(AUDIO_DIR, f'{cand_id}.wav'))
        src_cue    = [time_to_samples(c) for c in src_obj['cue'][1]]
        cand_cue   = [time_to_samples(c) for c in cand_obj['cue'][0]]
        mixability = estimate_mixability_by_nn(src_audio[: , src_cue[0]:src_cue[1]],
                                               cand_audio[:, cand_cue[0]:cand_cue[1]])        
        if mixability > best['mixability']:
            best['id']         = cand_id
            best['mixability'] = mixability
            
    return best['id']
            

def match_pair(src_id, cand_ids, n_sample=None, match_type=None): # all, nn, rule, None
    
    tgt_id = None
    
    
    if match_type in ['all', 'rule']:
        cand_ids = match_pair_by_rule(src_id, cand_ids)
        if len(cand_ids) == 0:
            return
        
    if match_type in ['all', 'nn']:
        tgt_id  = match_pair_by_nn(src_id, cand_ids, n_sample=n_sample)
        
    if tgt_id == None:
        tgt_id  = random_samples(cand_ids, 1)[0]
    
    return (src_id, tgt_id)

            