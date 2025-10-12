import sys
import json
import mne
import os
import os.path as op
import subprocess as sp
import numpy as np
import pandas as pd
from utilities import files
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pylab as plt

# parsing command line arguments
try:
    subject_idx = int(sys.argv[1])
except:
    print("incorrect subject index")
    sys.exit()

try:
    json_file = sys.argv[2]
    print("USING:", json_file)
except:
    json_file = "settings.json"
    print("USING:", json_file)

# opening a json file
with open(json_file) as pipeline_file:
    parameters = json.load(pipeline_file)

path = parameters["dataset_path"]
sfreq = parameters["downsample_dataset"]
der_path = op.join(path, "derivatives_v2")
files.make_folder(der_path)
proc_path = op.join(der_path, "processed")
files.make_folder(proc_path)

df = pd.read_csv(op.join(path, 'data_v2', 'GOGO_Demographics_2025_COMO.csv'))
if subject_idx<len(df):
    subject_id = df.iat[subject_idx, df.columns.get_loc("ParticipantID")]
else:
    df2 = pd.read_csv(op.join(path, 'data_v2', 'GOGO_Demographics_2025_Driving.csv'))
    subject_id = df2.iat[subject_idx-len(df), df2.columns.get_loc("ParticipantID")]

print("ID:", subject_id)

sub_path = op.join(proc_path, subject_id)

raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
raw_paths.sort()
raw_path = raw_paths[0]

event_paths = files.get_files(sub_path, subject_id, "-eve.fif")[2]
event_paths.sort()
event_path = event_paths[0]

ica_paths = files.get_files(sub_path, subject_id, "-ica.fif")[2]
ica_paths.sort()
ica_path = ica_paths[0]

ica_json_file = op.join(
    sub_path,
    "{}-ICA_to_reject.json".format(subject_id)
)


print("SUBJ: {}".format(subject_id))
print("INPUT RAW FILE:", raw_path.split(os.sep)[-1])
print("INPUT EVENT FILE:", event_path.split(os.sep)[-1])
print("INPUT ICA FILE:", ica_path.split(os.sep)[-1])
print("INPUT JSON FILE", ica_json_file.split(os.sep)[-1])

raw = mne.io.read_raw_fif(
    raw_path, preload=True, verbose=False
)

events = mne.read_events(event_path)

ica = mne.preprocessing.read_ica(
    ica_path, verbose=False
)

raw.crop(
    tmin=raw.times[events[0,0]],
    tmax=raw.times[events[-1,0]]
)
raw.filter(1,20, verbose=False)
raw.close()

sp.Popen(
    ["mousepad", str(ica_json_file)],
    stdout=sp.DEVNULL,
    stderr=sp.DEVNULL
)
print('')

title_ = "sub:{}, file: {}".format(subject_id, ica_path.split(os.sep)[-1])

ica.plot_components(inst=raw, picks=np.arange(25), show=False, title=title_)

ica.plot_sources(inst=raw, picks=np.arange(25), show=False, title=title_)

plt.show()

