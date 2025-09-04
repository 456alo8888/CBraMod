import os
import subprocess

# Đường dẫn thư mục gốc EEG2100
INPUT_DIR = "/mnt/disk1/aiotlab/hieupc/CBraMod/EEG2100"
NK2EDF_PATH = "/mnt/disk1/aiotlab/hieupc/CBraMod/nk2edf"
OUTPUT_DIR = os.path.join(INPUT_DIR, "edf_files")

# Tạo thư mục output nếu chưa có
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Bước 1: rename file extension về thường
for fname in os.listdir(INPUT_DIR):
    old_path = os.path.join(INPUT_DIR, fname)
    if fname.endswith(".EEG"):
        new_path = os.path.join(INPUT_DIR, fname[:-4] + ".eeg")
        os.rename(old_path, new_path)
    elif fname.endswith(".PNT"):
        new_path = os.path.join(INPUT_DIR, fname[:-4] + ".pnt")
        os.rename(old_path, new_path)
    elif fname.endswith(".LOG"):
        new_path = os.path.join(INPUT_DIR, fname[:-4] + ".log")
        os.rename(old_path, new_path)

# Bước 2: convert từng file EEG
for fname in os.listdir(INPUT_DIR):
    if fname.endswith(".eeg"):
        base = fname[:-4]
        eeg_path = os.path.join(INPUT_DIR, base + ".eeg")
        pnt_path = os.path.join(INPUT_DIR, base + ".pnt")
        log_path = os.path.join(INPUT_DIR, base + ".log")

        print(f"[+] Processing {base}")

        if os.path.exists(pnt_path) and os.path.exists(log_path):
            # Đủ bộ ba -> EDF+
            cmd = [NK2EDF_PATH, eeg_path]
        else:
            # Thiếu -> EDF thường
            cmd = [NK2EDF_PATH, "-no-annotations", eeg_path]

        try:
            subprocess.run(cmd, check=True)
            # Move tất cả file .edf sinh ra vào output dir
            for f in os.listdir(INPUT_DIR):
                if f.startswith(base) and f.endswith(".edf"):
                    os.rename(os.path.join(INPUT_DIR, f), os.path.join(OUTPUT_DIR, f))
                    print(f"    -> Converted: {f}")
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Error converting {base}: {e}")
