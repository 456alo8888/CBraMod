import os
import random

import mne
import numpy as np
from tqdm import tqdm
import pickle
import lmdb


selected_channels = {
    '01_tcp_ar': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ],
    '02_tcp_le': [
            'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE', 'EEG C4-LE', 'EEG P3-LE',
            'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE', 'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE',
            'EEG T5-LE', 'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'
    ],
    '03_tcp_ar_a': [
            'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF',
            'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF',
            'EEG T5-REF', 'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'
    ]
}

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


#遍历文件夹
def iter_files(rootDir):
    #遍历根目录
    file_path_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            # print(file_name)
            file_path_list.append(file_name)
    return file_path_list

def preprocessing_recording(file_path, file_key_list: list, db: lmdb.open):
    raw = mne.io.read_raw_edf(file_path, preload=True)

    if '02_tcp_le' in file_path:
        montage_type = '02_tcp_le'
    elif '01_tcp_ar' in file_path:
        montage_type = '01_tcp_ar'
    elif '03_tcp_ar_a' in file_path:
        montage_type = '03_tcp_ar_a'

    if not montage_type:
        return

    channels_to_pick = selected_channels[montage_type]
    # Ensure all required channels are present before picking
    if not all(ch in raw.info['ch_names'] for ch in channels_to_pick):
        return
    raw.pick_channels(channels_to_pick, ordered=True)

    raw.resample(200)
    raw.filter(l_freq=0.3, h_freq=75)
    raw.notch_filter((60))
    # Use get_data() for a more direct conversion to a NumPy array
    eeg_array = raw.get_data(units='uV').T
    points, chs = eeg_array.shape
    if points < 300 * 200:
        return
    a = points % (30 * 200)

    patch_size = 30*200
    overlap_size = patch_size // 2 
    step_size = patch_size - overlap_size 

    start_idx = 60*200 
    end_idx = points - a - 60*200
    usable_length = end_idx - start_idx

    num_patches = (usable_length - patch_size)// step_size + 1 
    patches = []

    for i in range(num_patches):
        patch_start = start_idx + i *step_size 
        patch_end = patch_start + patch_size
        patch = eeg_array[patch_start:patch_end, :]
        #Reshape to (30 , 200 , channels)
        patch = patch.reshape(30 , 200 , chs)

        #Transpose to (channels , 30, 200)
        patch = patch.transpose(2, 0, 1)
        patches.append(patch)
    patches = np.array(patches)
    print(f"Created {len(patches)} patches of shape {patches.shape}")
    # eeg_array = eeg_array[60 * 200:-(a+60 * 200), :]
    # print(eeg_array.shape)
    # eeg_array = eeg_array.reshape(-1, 30, 200, chs)
    # eeg_array = eeg_array.transpose(0, 3, 1, 2)
    # print(eeg_array.shap
    file_name = file_path.split('/')[-1][:-4]

    for i, sample in enumerate(patches):
        # print(i, sample.shape)
        if np.max(np.abs(sample)) < 100:
            sample_key = f'{file_name}_{i}'
            print(sample_key)
            file_key_list.append(sample_key)
            txn = db.begin(write=True)
            txn.put(key=sample_key.encode(), value=pickle.dumps(sample))
            txn.commit()

if __name__ == '__main__':
    setup_seed(1)
    file_path_list = iter_files('path...')

    file_path_list = sorted(file_path_list)
    random.shuffle(file_path_list)
    # print(file_path_list)
    db = lmdb.open(r'path...', map_size=1649267441664)
    file_key_list = []
    for file_path in tqdm(file_path_list):
        preprocessing_recording(file_path, file_key_list, db)

    txn = db.begin(write=True)
    txn.put(key='__keys__'.encode(), value=pickle.dumps(file_key_list))
    txn.commit()
    db.close()
