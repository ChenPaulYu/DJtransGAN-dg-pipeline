import msaf
from joblib import Memory

from pipeline.config  import settings
from pipeline.feature import estimate_beat, correct_boundary

memory = Memory(settings.CACHE_DIR, verbose=1)


@memory.cache
def segment_structure(audio):
    try:
        boundary, label = msaf.process(audio, 
                                         feature='cqt', 
                                         boundaries_id='sf', 
                                         labels_id='fmc2d')
        downbeat = estimate_beat(audio)[-1]
        boundary = correct_boundary(boundary, downbeat)
        
    except Exception as e:
        print('segment error: ', e)
        boundary, label = [], []
    return boundary, label