import sys
import json
from mne import read_epochs, set_log_level
import matplotlib
matplotlib.use('Agg')
from matplotlib import colors
import os.path as op
import pandas as pd
from utilities import files
import matplotlib.pylab as plt
from autoreject import compute_thresholds
import numpy as np

set_log_level(verbose=False)

def run(subject_id, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]

    der_path = op.join(path, "derivatives_v2")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)

    qc_folder = op.join(sub_path, "QC")

    epo_paths = files.get_files(sub_path, subject_id, "-epo.fif")[2]
    epo_paths.sort()

    cmap = colors.ListedColormap(["#FFFFFF", "#CFEEFA", "#FFDE00", "#FF9900", "#FF0000", "#000000"])
    boundaries = [-0.9, -0.1, 1.1, 10, 100, 1000, 10000]
    norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)

    # for epo in epo_paths:
    for epo in epo_paths:
        print("INPUT FILE:", epo)
        epochs = read_epochs(epo, verbose=False)

        ch_thr = compute_thresholds(
            epochs,
            random_state=42,
            method="bayesian_optimization",
            verbose="progressbar",
            n_jobs=-1,
            augment=False
        )
        # save the thresholds in JSON
        ch_list = list(ch_thr.keys())
        ch_list.sort()
        results = np.zeros((len(ch_list), len(epochs)))
        results = results - 1
        for ix, ch in enumerate(ch_list):
            thr = ch_thr[ch]
            ch_tr = epochs.copy().pick_channels([ch]).get_data()
            res = [np.where(ch_tr[i][0] > thr)[0].shape[0] for i in range(len(epochs))]
            res = np.array(res)
            results[ix, :] = res
        name = subject_id
        npy_path = op.join(qc_folder, name + ".npy")
        np.save(npy_path, results)
        img_path = op.join(qc_folder, name + "-epo-QC.png")
        print(results[:15, :15])
        print(np.min(results), np.max(results))
        print(np.unique(results))

        plt.rcParams.update({'font.size': 5})
        f, ax = plt.subplots(
            figsize=(20, 20),
            dpi=200
        )

        im = ax.imshow(
            results,
            aspect="auto",
            cmap=cmap,
            interpolation="none",
            norm=norm
        )
        f.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
        ax.set_xlabel("Trials")
        ax.set_ylabel("Channels")
        ax.set_xticks(list(range(len(epochs))))
        ax.set_xticklabels([str(i) for i in range(1, len(epochs) + 1)])
        ax.set_yticks(list(range(len(ch_list))))
        ax.set_yticklabels(ch_list)
        ax.grid(color='w', linestyle='-', linewidth=0.2)
        ax.set_title(name)
        plt.savefig(
            img_path,
            bbox_inches="tight"
        )
        plt.close("all")


if __name__=='__main__':
    json_file = "settings.json"
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)
    path = parameters["dataset_path"]

    df = pd.read_csv(op.join(path, 'data_v2', 'GOGO_Demographics_2025_COMO.csv'))
    for subject_id, group in df.loc[:, ["ParticipantID", "Status"]].itertuples(index=False, name=None):
        run(subject_id, json_file)

    df = pd.read_csv(op.join(path, 'data_v2', 'GOGO_Demographics_2025_Driving.csv'))
    for subject_id, group in df.loc[:, ["ParticipantID", "Status"]].itertuples(index=False, name=None):
        run(subject_id, json_file)