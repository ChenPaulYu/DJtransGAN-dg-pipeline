from pipeline.feature.utils     import *
from pipeline.feature.beat      import *
from pipeline.feature.key       import *
from pipeline.feature.structure import *
from pipeline.feature.separate  import *


def estimate_feature(audio_path):
    estimated = {}
    estimated['beat']      = estimate_beat(audio_path)
    estimated['key']       = estimate_key(audio_path)
    estimated['structure'] = segment_structure(audio_path)
    return estimated