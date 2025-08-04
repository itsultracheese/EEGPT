import tqdm
import torch
import os
import mne
import numpy as np
from tqdm import tqdm

input_folder = "/mnt/s3-data2/amiftakhova/OCD/OCD/"
save_folder = "/mnt/s3-data2/amiftakhova/ocd_eegpt/"

all_chans = ['Fp1', 'Fp2', 'F3', 'Fz', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'Cz', 'C4', 'T5', 'T6', 'P3', 'Pz', 'P4', 'O1', 'O2']

s_freq = 256
fmin = 0
fmax = 30
tmin = 0.0
tmax = 20.0
overlap = 15.0

folders = [x for x in os.listdir(input_folder) if os.path.isdir(input_folder + x)]
inside_folders = []
for fldr in folders:
    ins = [fldr + '/' + x for x in os.listdir(input_folder + fldr) if os.path.isdir(input_folder + fldr + '/' + x)]
    if len(ins) > 0:
        inside_folders.extend(ins)
    else:
        inside_folders.append(fldr)
fldr2label = {inside_folders[i]: i for i in range(len(inside_folders))}

label2class = {
    4: 0, 5: 0, 6: 0, # anxiety
    7: 1, 8: 1, 9: 1, # bipolar
    10: 2, # control
    11: 3, 12: 3, 13: 3, # depression
    14: 4, # personality disorder,
    0: 5, 1: 5, 2: 5, 3: 5# stress
}

class2name = {0: 'anxiety', 1: 'bipolar', 2: 'control', 3: 'depression', 4: 'personality disorder', 5: 'stress'}
name2class = {v: k for k, v in class2name.items()}

og_files = []
zg_files = []
og_labels = []
zg_labels = []
for fldr in inside_folders:
    pth =  input_folder + fldr
    og_pths = [x for x in os.listdir(pth) if x.lower().endswith('.edf') and ('og.' in x.lower() or '_ог.' in x.lower() or ' ог.' in x.lower() or 'eo.' in x.lower() or '_eo' in x.lower() or '_of' in x.lower())]
    zg_pths = [x for x in os.listdir(pth) if x.lower().endswith('.edf') and ('zg.' in x.lower() or 'зог.' in x.lower() or 'зг.' in x.lower() or 'ec.' in x.lower() or 'fon.' in x.lower() or '_ec' in x.lower() or 'eс.' in x.lower())]
    left = [x for x in os.listdir(pth) if x not in og_pths and x not in zg_pths]
    if len(left) > 0:
        print(pth, left)
    for f in og_pths:
        og_files.append(pth + '/' + f)
        og_labels.append(fldr2label[fldr])
    for f in zg_pths:
        zg_files.append(pth + '/' + f)
        zg_labels.append(fldr2label[fldr])

print(f'Number of files for open eyes: {len(og_files)}')
print(f'Number of files for closed eyes: {len(zg_files)}')

lengths_og = []
lengths_zg = []
for i in tqdm(range(len(og_files))):
    og_file = og_files[i]
    zg_file = zg_files[i]
    try:
        og_sample = mne.io.read_raw_edf(og_file, verbose=False)
        og_len = 1.0 * len(og_sample) / og_sample.info['sfreq']
        zg_sample = mne.io.read_raw_edf(zg_file, verbose=False)
        zg_len = 1.0 * len(zg_sample) / zg_sample.info['sfreq']
        lengths_og.append(og_len)
        lengths_zg.append(zg_len)
    except Exception as e:
        print(f, e)
        lengths_og.append(0)
        lengths_zg.append(0)
ids = np.argwhere(np.array(lengths_zg) >= 20.0).flatten()

to_skip = ['BORUTTO_JANNA_VLADIMIROVNA', 'Kutuz_f23_contr', 'MANUILOVA_ELENA_55', 'Martinenko_m45', 'Skopincev_20', 'FiAV_m50']

subj_id = 0
for i in tqdm(ids):
    path = zg_files[i]
    if any([x in path for x in to_skip]):
        continue
    sample = mne.io.read_raw_edf(path, verbose=False, preload=True)
    sample = sample.resample(s_freq, verbose=False)

    sample = sample.filter(l_freq=1, h_freq=30, method='iir', verbose=False)
    channels = sample.ch_names
    to_drop = channels[19:]

    new_idx = []
    skip = False
    for ch in all_chans:
        found = False
        for k in range(19):
            if ch in channels[k]:
                new_idx.append(k)
                found = True
                break
        if not found:
            skip = True
            break
    sample = sample.pick(np.array(channels)[new_idx])
    if len(sample) / s_freq > 60:
        sample = sample.crop(tmin=0.0, tmax=60.0)
    events = mne.make_fixed_length_events(sample, duration=tmax, overlap=overlap)
    epochs = mne.Epochs(sample, events, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
    data = epochs.get_data()
    label = label2class[zg_labels[i]]
    for j in range(data.shape[0]):
        x = torch.tensor(data[j])
        y = label
        spath = save_folder + f'{y}/'
        os.makedirs(spath, exist_ok=True)
        spath = spath + f'{j}.sub{subj_id}'
        torch.save(x, spath)
    subj_id += 1

# for sub in [2,3,4,5,6,7,9,11]:
#     path = "erp-based-brain-computer-interface-recordings-1.0.0/files/s{:02d}".format(sub)
#     for file in os.listdir(path):
#         if not file.endswith(".edf"):continue
#         raw = mne.io.read_raw_edf(os.path.join(path, file))
#         raw.pick_channels(all_chans)
        
#         events, event_id = mne.events_from_annotations(raw)
        
#         event_map = {}
#         tgt = None
#         for k,v in event_id.items():
#             if k[0:4]=='#Tgt':
#                 tgt = k[4]
#             event_map[v] = k
#         # assert event_map[1][0:4]=='#Tgt' and event_map[2]=='#end' and event_map[3]=='#start', event_map
#         assert tgt is not None
#         epochs = mne.Epochs(raw, events, event_id=event_id, tmin = tmin, tmax=tmax,event_repeated='drop', preload=True, proj=False)#,event_repeated='drop',reject_by_annotation=True)
#         epochs.filter(fmin, fmax,method = 'iir')
#         epochs.resample(256)
#         stims = [x[2] for x in epochs.events]
#         # print(stims)
#         data = epochs.get_data()
#         for i,(d,t) in tqdm.tqdm(enumerate(zip(data, stims))):
#             t = event_map[t]
#             if t.startswith('#Tgt') or t.startswith('#end') or t.startswith('#start') or t[0]=='#':
#                 continue
#             label = 1 if tgt in t else 0
#             # -- save
#             x = torch.tensor(d*1e3)
#             y = label
#             spath = dataset_fold+f'{y}/'
#             os.makedirs(path,exist_ok=True)
#             spath = spath + f'{i}.sub{sub}'
#             torch.save(x, spath)
