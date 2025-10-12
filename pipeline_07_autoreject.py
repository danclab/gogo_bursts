import sys
import json
import os.path as op
from os import sep
import numpy as np
import pandas as pd
from mne import read_epochs, set_log_level
from utilities import files
from autoreject import AutoReject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

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
    if not op.exists(sub_path):
        return None

    qc_folder = op.join(sub_path, "QC")

    epo_path = op.join(sub_path, f'{subject_id}-epo.fif')
    epochs = read_epochs(epo_path, verbose=False, preload=True)
    print("AMOUNT OF EPOCHS:", len(epochs))

    behav_df = pd.read_csv(op.join(sub_path, f'{subject_id}-epo-behav.csv'))

    name = subject_id

    ar = AutoReject(
        consensus=np.linspace(0, 1.0, 27),
        n_interpolate=np.array([1, 4, 32]),
        thresh_method="bayesian_optimization",
        cv=10,
        n_jobs=-1,
        random_state=42,
        verbose="progressbar"
    )
    ar.fit(epochs)

    ar_fname = op.join(
        qc_folder,
        "{}-autoreject.h5".format(name)
    )
    ar.save(ar_fname, overwrite=True)
    epochs_ar, rej_log = ar.transform(epochs, return_log=True)

    rej_log.plot(show=False)
    plt.savefig(op.join(qc_folder, "{}-autoreject-log.png".format(name)))
    plt.close("all")

    # Good/bad flags correspond to the ORIGINAL epochs order
    good_epoch = ~rej_log.bad_epochs

    ev = epochs.events
    ev_code = ev[:, 2]
    ev_samp = ev[:, 0]
    eid = epochs.event_id  # {'STIM/LONG':..., 'RESP/LONG':..., ...}

    STIM_LONG = eid['STIM/LONG']
    STIM_SHORT = eid['STIM/SHORT']
    RESP_LONG = eid['RESP/LONG']
    RESP_SHORT = eid['RESP/SHORT']

    is_stim = np.isin(ev_code, [STIM_LONG, STIM_SHORT])
    is_resp = np.isin(ev_code, [RESP_LONG, RESP_SHORT])

    # Trial index over STIMs only (0-based, in temporal order)
    stim_samples = ev_samp[is_stim]
    n_trials = len(stim_samples)
    sample_to_trial = {int(s): i for i, s in enumerate(stim_samples)}

    # Pair each STIM with the first subsequent RESP within condition
    def _pair(stim_samp, resp_samp):
        stim_samp = np.asarray(stim_samp)
        resp_samp = np.sort(np.asarray(resp_samp))
        j = 0
        mapping = {}
        for s in stim_samp:
            while j < len(resp_samp) and resp_samp[j] <= s:
                j += 1
            if j < len(resp_samp):
                mapping[int(resp_samp[j])] = int(sample_to_trial[int(s)])
                j += 1
        return mapping

    resp_map = {}
    resp_map.update(_pair(ev_samp[ev_code == STIM_LONG], ev_samp[ev_code == RESP_LONG]))
    resp_map.update(_pair(ev_samp[ev_code == STIM_SHORT], ev_samp[ev_code == RESP_SHORT]))
    # resp_map: RESP sample -> trial index of its paired STIM

    # Masks after AutoReject
    stim_kept_ar = np.zeros(n_trials, dtype=bool)
    resp_kept_ar = np.zeros(n_trials, dtype=bool)

    for samp, code, good in zip(ev_samp, ev_code, good_epoch):
        if not good:
            continue
        if code in (STIM_LONG, STIM_SHORT):
            stim_kept_ar[sample_to_trial[int(samp)]] = True
        elif code in (RESP_LONG, RESP_SHORT):
            t = resp_map.get(int(samp), None)
            if t is not None:
                resp_kept_ar[t] = True

    # --- Align and UPDATE behavior (behav_df has one row per STIM in trial order) ---
    assert len(behav_df) == n_trials, (
        f"Behavior rows ({len(behav_df)}) don't match STIM trials ({n_trials}). "
        "Ensure epo-behav was saved before any epoch rejection."
    )

    # Drop STIM trials rejected by AutoReject
    behav_df = behav_df.loc[stim_kept_ar].reset_index(drop=True)

    # Update/overwrite the kept flags to reflect AR results
    # (for the surviving STIM trials, provide parallel RESP-kept flags)
    behav_df['stim_kept'] = True
    behav_df['resp_kept'] = resp_kept_ar[stim_kept_ar].astype(bool)

    # --- Save outputs as before ---
    cleaned = op.join(sub_path, "autoreject-" + epo_path.split(sep)[-1])
    epochs_ar.save(cleaned, overwrite=True)
    print("CLEANED EPOCHS SAVED:", cleaned)

    behav_path = op.join(sub_path, f"autoreject-{subject_id}-epo-behav.csv")
    behav_df.to_csv(behav_path, index=False)
    print("UPDATED BEHAV SAVED:", behav_path)


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