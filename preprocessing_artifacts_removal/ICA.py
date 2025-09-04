import os 
import mne 
from mne.preprocessing import ICA, corrmap, create_ecg_epochs , create_eog_epochs
import numpy as np


import os
import pickle
import lmdb
import mne
import numpy as np



EDF_FOLDER = "/mnt/disk1/aiotlab/hieupc/CBraMod/EEG2100/edf_files"

EEG_INCLUDE = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4',
               'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
               'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ']
EEG_EXCLUDE = ['SPO2', 'ETCO2', 'PULSE', 'CO2WAVE',
               'DC03', 'DC04', 'DC05', 'DC06']


# def ICA_for_file(file_path):
#     # Đọc toàn bộ dữ liệu
#     raw_full = mne.io.read_raw_edf(file_path, preload=True)

#     # Lấy danh sách kênh EEG hợp lệ
#     eeg_channels = [
#         ch for ch in raw_full.ch_names
#         if any(eeg_name in ch.upper() for eeg_name in EEG_INCLUDE)
#         and ch.upper() not in EEG_EXCLUDE
#     ]
#     raw_full.pick(eeg_channels)

#     # Tạo bản copy để fit ICA (crop nếu quá dài)
#     raw_for_ica = raw_full.copy()
#     duration = raw_for_ica.times[-1]  # thời gian file (giây)
#     tmax = min(duration, 180)  # chỉ lấy tối đa 180s để fit
#     if duration > 180:
#         raw_for_ica.crop(tmax=tmax)

#     # Lọc tín hiệu (0.5 Hz high-pass để loại bỏ drift DC)
#     filt_raw = raw_for_ica.filter(l_freq=0.5, h_freq=None)

#     # Fit ICA
#     ica = ICA(n_components=30, max_iter="auto", random_state=42)
#     ica.fit(filt_raw)

#     # Tìm EOG artifact
#     eog_indices, _ = ica.find_bads_eog(raw_for_ica, ch_name='Fp1', threshold=0.9)

#     # Tìm ECG artifact
#     ecg_indices, _ = ica.find_bads_ecg(raw_for_ica, ch_name='C3', method='correlation', threshold=0.9)

#     # Loại bỏ artifact
#     ica.exclude = ecg_indices + eog_indices

#     # Áp dụng ICA lên toàn bộ dữ liệu (full length)
#     reconst_raw = raw_full.copy()
#     ica.apply(reconst_raw)

#     return reconst_raw

def ICA_for_file(file_path):
    # Đọc toàn bộ dữ liệu
    raw_full = mne.io.read_raw_edf(file_path, preload=True)

    # Lọc kênh EEG hợp lệ
    eeg_channels = [
        ch for ch in raw_full.ch_names
        if any(eeg_name in ch.upper() for eeg_name in EEG_INCLUDE)
        and ch.upper() not in EEG_EXCLUDE
    ]
    raw_full.pick(eeg_channels)

    # Tạo bản copy để fit ICA trên 60s đầu
    raw_for_ica = raw_full.copy()
    duration = raw_for_ica.times[-1]
    tmax = min(duration, 60)  # chỉ lấy tối đa 60s
    raw_for_ica.crop(tmax=tmax)

    # Lọc tín hiệu (high-pass 0.5 Hz để loại bỏ drift)
    filt_raw = raw_for_ica.filter(l_freq=0.5, h_freq=None)

    # Fit ICA
    ica = ICA(n_components=10, max_iter="auto", random_state=42)
    ica.fit(filt_raw)

    # Tìm EOG artifact
    eog_indices, _ = ica.find_bads_eog(raw_for_ica, ch_name='Fp1', threshold=0.9)

    # Tìm ECG artifact
    ecg_indices, _ = ica.find_bads_ecg(raw_for_ica, ch_name='C3', method='correlation', threshold=0.9)

    # Loại bỏ artifact
    ica.exclude = ecg_indices + eog_indices

    # Áp dụng ICA lên toàn bộ dữ liệu
    reconst_raw = raw_full.copy()
    ica.apply(reconst_raw)

    return reconst_raw



def read_edf_cut_duration(raw):
    eeg_channels = [
        ch for ch in raw.ch_names
        if any(eeg_name in ch.upper() for eeg_name in EEG_INCLUDE)
        and ch.upper() not in EEG_EXCLUDE
    ]
    raw.pick(eeg_channels)

    data, times = raw.get_data(return_times=True)
    sfreq = int(raw.info['sfreq'])

    segment_len = int(sfreq * 0.1)
    n_segments = data.shape[1] // segment_len

    valid_segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len
        seg = data[:, start:end]
        seg_uV = seg * 1e6  # V -> µV

        if np.max(np.abs(seg_uV)) <= 100:
            valid_segments.append(seg)

    if valid_segments:
        filtered_data = np.concatenate(valid_segments, axis=1)
    else:
        raise ValueError("No valid segment found, all segments > 100 µV")

    info = raw.info.copy()
    raw_filtered = mne.io.RawArray(filtered_data, info)
    duration_seconds = filtered_data.shape[1] / sfreq

    return duration_seconds

def filter_raw(raw):
    eeg_channels = [
        ch for ch in raw.ch_names
        if any(eeg_name in ch.upper() for eeg_name in EEG_INCLUDE)
        and ch.upper() not in EEG_EXCLUDE
    ]
    raw.pick(eeg_channels)

    data, times = raw.get_data(return_times=True)
    sfreq = int(raw.info['sfreq'])
    segment_len = int(sfreq * 0.2)
    n_segments = data.shape[1] // segment_len

    valid_segments = []
    for i in range(n_segments):
        start = i * segment_len
        end = start + segment_len
        seg = data[:, start:end] * 1e6  # V → µV
        if np.max(np.abs(seg)) <= 100:
            valid_segments.append(seg / 1e6)  # chuyển lại V

    if not valid_segments:
        return None

    filtered_data = np.concatenate(valid_segments, axis=1)
    return filtered_data, raw.info.copy()

def create_lmdb_from_folder(edf_folder, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1 << 40)  # 1 TB max
    keys = []
    idx = 0

    with env.begin(write=True) as txn:
        for fname in os.listdir(edf_folder):
            if not fname.lower().endswith(".edf"):
                continue
            file_path = os.path.join(edf_folder, fname)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            result = filter_raw(raw)
            if result is None:
                continue
            filtered_data, info = result
            key = f"{fname}_{idx}"
            txn.put(key.encode(), pickle.dumps((filtered_data, info)))
            keys.append(key)
            idx += 1

        # Lưu danh sách key
        txn.put("__keys__".encode(), pickle.dumps(keys))

    env.close()
    print(f"Saved {len(keys)} samples to LMDB at {lmdb_path}")


def create_lmdb_from_folder(edf_folder, lmdb_path):
    env = lmdb.open(lmdb_path, map_size=1 << 40)  # 1 TB max
    keys = []
    idx = 0

    with env.begin(write=True) as txn:
        for fname in os.listdir(edf_folder):
            if not fname.lower().endswith(".edf"):
                continue
            file_path = os.path.join(edf_folder, fname)

            reconst_raw = ICA_for_file(file_path)
            # raw = mne.io.read_raw_edf(file_path, preload=True)
            result = filter_raw(reconst_raw)
            if result is None:
                continue
            filtered_data, info = result
            key = f"{fname}_{idx}"
            txn.put(key.encode(), pickle.dumps((filtered_data, info)))
            keys.append(key)
            idx += 1

        # Lưu danh sách key
        txn.put("__keys__".encode(), pickle.dumps(keys))

    env.close()
    print(f"Saved {len(keys)} samples to LMDB at {lmdb_path}")

def get_lmdb_size(lmdb_path):
    total_bytes = sum(
        os.path.getsize(os.path.join(lmdb_path, f))
        for f in os.listdir(lmdb_path)
    )
    print(f"LMDB size: {total_bytes / (1024**3):.2f} GB")

# Example usage
# edf_folder = "/mnt/disk1/aiotlab/hieupc/CBraMod/EEG2100/edf_files"
# lmdb_path = "/mnt/disk1/aiotlab/hieupc/filtered_eeg.lmdb"

# create_lmdb_from_folder(edf_folder, lmdb_path)
# get_lmdb_size(lmdb_path)



if __name__ == "__main__":

    edf_folder = "/mnt/disk1/aiotlab/hieupc/CBraMod/EEG2100/edf_files"
    lmdb_path = "/mnt/disk1/aiotlab/hieupc/filtered_eeg.lmdb"

    create_lmdb_from_folder(edf_folder, lmdb_path)
    get_lmdb_size(lmdb_path)
    # total_duration = 0
    # for fname in os.listdir(EDF_FOLDER):
    #     if fname.lower().endswith(".edf"):
    #         file_path = os.path.join(EDF_FOLDER, fname)
    #         print(f"Processing {file_path} ...")

    #         try:
    #             # 1. Áp dụng ICA
    #             raw_after_ica = ICA_for_file(file_path)
    #             # raw_full = mne.io.read_raw_edf(file_path, preload=True)

    #             # 2. Tính thời gian còn lại
    #             dur = read_edf_cut_duration(raw_after_ica)
    #             # dur = read_edf_cut_duration(raw_full)
    #             total_duration += dur

    #             print(f"  Remaining: {dur:.2f} secs")
    #         except Exception as e:
    #             print(f"  Error processing {fname}: {e}")

    # print(f"\nTotal remaining duration: {total_duration:.2f} secs")