cd /mnt/disk1/aiotlab/hieupc/CBraMod

for f in /mnt/disk1/aiotlab/hieupc/CBraMod/EEG2100/*.EEG; do
    ./nk2edf "$f"
done