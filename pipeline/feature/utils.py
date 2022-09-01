import numpy as np
from pipeline.utils import find_nearest

KEY_TABLE = {
    'minor': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'], 
    'major': ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
}


def keylabel_to_keynum(key_label):
    key, style  = key_label.split(' ') 
    use_table   = KEY_TABLE[style]
    return (-1, None) if key not in use_table else (use_table.index(key), style)

def keynum_to_keylabel(key_data):
    key_num, style = key_data
    use_table      = KEY_TABLE[style]
    return f'{use_table[key_num]} {style}'

def filter_beat(beat, downbeat, sig_d):
    begin = np.where(beat == downbeat[0])[0][0]
    end   = np.where(beat == downbeat[-1])[0][0] + sig_d
    return beat[begin:end]

def correct_boundary(boundary, downbeat):
    return np.unique(np.fromiter((downbeat[find_nearest(downbeat, bound)] for bound in boundary), boundary.dtype))

