from madmom.features.key import key_prediction_to_label, CNNKeyRecognitionProcessor
from joblib import Memory

from pipeline.config  import settings
from pipeline.feature import keylabel_to_keynum

memory = Memory(settings.CACHE_DIR, verbose=1)


@memory.cache
def estimate_key(audio):
    proc = CNNKeyRecognitionProcessor()
    try:
        res = proc(audio)  
        key = key_prediction_to_label(res)
        return keylabel_to_keynum(key)
    except:
        return (-1, None)
    
    
