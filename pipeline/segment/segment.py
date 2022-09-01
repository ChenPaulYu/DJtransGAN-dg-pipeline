import os
from joblib import Memory

from pipeline.config  import settings
from pipeline.feature import estimate_feature, estimate_stem, estimate_sources, get_separator
from pipeline.feature import load_audio
from pipeline.utils   import find_index, time_to_samples

memory = Memory(settings.CACHE_DIR, verbose=1)


def get_boundary_pair(boundary, downbeat):
    boundary_pair = []
    for i, b in enumerate(boundary):
        b_idx = find_index(downbeat, b)
        for b_next in boundary[i+1:]:
            b_next_idx = find_index(downbeat, b_next)
            if b_next_idx - b_idx > settings.SEG_BAR:
                boundary_pair.append((b, b_next))
                break
    return boundary_pair



def get_segment(audio_path, separator):
    
    out_dict  = {}
    segments  = []
    audio     = load_audio(audio_path) 
    stem      = estimate_stem(audio_path, separator)
    feature   = estimate_feature(audio_path)
    boundary  = feature['structure'][0]
    sig, bpm, beat, downbeat = feature['beat']
    
    out_dict['origin'] = os.path.basename(audio_path)
    out_dict['id']     = out_dict['origin'].split('.')[0]
    out_dict['key']    = feature['key'][0]
    out_dict['bpm']    = bpm
    out_dict['sig']    = sig  

    for i, pair in enumerate(get_boundary_pair(boundary, downbeat)):
        estimated_obj   = {'id': out_dict['id'], 'key': out_dict['key'], 'bpm': out_dict['bpm']}   
        dbs             = downbeat[find_index(downbeat, pair[0]):find_index(downbeat, pair[1])]
        bs              = beat[find_index(beat, pair[0]):find_index(beat, pair[1])-3]
        
        if len(dbs) > settings.CUE_BAR:
        
            estimated_audio     = audio[:, time_to_samples(dbs[0]):time_to_samples(dbs[-1])]
            estimated_stem      = stem[:, :, time_to_samples(dbs[0]):time_to_samples(dbs[-1])]
                        
            estimated_beat      = bs-bs[0]
            estimated_downbeat  = dbs-dbs[0]
            
    
            estimated_source1   = estimate_sources(stem[:, :, time_to_samples(estimated_downbeat[0]): 
                                                              time_to_samples(estimated_downbeat[settings.CUE_BAR])])
            estimated_source2   = estimate_sources(stem[:, :, time_to_samples(estimated_downbeat[-settings.CUE_BAR-1]): 
                                                              time_to_samples(estimated_downbeat[-1])])            
            
            estimated_obj['seg_id']   = f'{out_dict["id"]}_{i}'
            estimated_obj['beat']     = estimated_beat
            estimated_obj['downbeat'] = estimated_downbeat 
            estimated_obj['source']   = [estimated_source1, estimated_source2]
            estimated_obj['cue']      = [[estimated_downbeat[0], estimated_downbeat[settings.CUE_BAR]], 
                                         [estimated_downbeat[-settings.CUE_BAR-1], estimated_downbeat[-1]]]
            
            segments.append({
                'seg_id': estimated_obj['seg_id'],
                'obj'   : estimated_obj, 
                'audio' : estimated_audio, 
                'info'  : out_dict
            })
            
    
    return segments