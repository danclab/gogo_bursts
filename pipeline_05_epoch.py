import sys
import json
import mne
import os.path as op
import pandas as pd
import numpy as np
from mne import events_from_annotations

from utilities import files


def run(subject_id, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    high_pass = parameters["high_pass_filter"]
    low_pass = parameters["low_pass_filter"]

    der_path = op.join(path, "derivatives_v2")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    print("ID:", subject_id)

    raw_fname = op.join(path, 'data_v2', f'{subject_id}_GOGO-raw.fif')
    if not op.exists(raw_fname):
        return None

    orig_raw = mne.io.read_raw_fif(raw_fname)
    _, event_id = events_from_annotations(orig_raw)

    sub_path = op.join(proc_path, subject_id)

    raw_path = op.join(sub_path, f'zapline-{subject_id}-raw.fif')

    ica_json_file = op.join(
        sub_path,
        "{}-ICA_to_reject.json".format(subject_id)
    )

    with open(ica_json_file) as ica_file:
        ica_files = json.load(ica_file)

    ica_key = f'{subject_id}-ica.fif'

    eve_path = op.join(sub_path, f'{subject_id}-eve.fif')

    # for (raw_path, ica_key, eve_path) in [raw_ica_eve[3]]:
    ica_path = op.join(
        sub_path,
        ica_key
    )

    print("INPUT RAW FILE:", raw_path)
    print("INPUT EVENT FILE:", eve_path)
    print("INPUT ICA FILE:", ica_path)

    ica_exc = ica_files[ica_key]

    events = mne.read_events(eve_path)

    ica = mne.preprocessing.read_ica(
        ica_path,
        verbose=False
    )

    raw = mne.io.read_raw_fif(
        raw_path,
        verbose=False,
        preload=True
    )

    raw = ica.apply(
        raw,
        exclude=ica_exc,
        verbose=False
    )
    raw = raw.pick_types(meg=True, eeg=False, ref_meg=True)

    raw.filter(
        l_freq=high_pass,
        h_freq=low_pass
    )

    # Trial-level behavior (one row per STIM, in STIM temporal order)
    behav_df = pd.read_csv(op.join(sub_path, f'{subject_id}-behav.csv'))

    # Combined epochs (STIM+RESP)
    epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2,
                        baseline=None, event_repeated='merge', preload=True)

    # ---- Build STIM order and STIM?RESP pairing (in raw "events" space) ----
    STIM_LONG = event_id['STIM/LONG']
    STIM_SHORT = event_id['STIM/SHORT']
    RESP_LONG = event_id['RESP/LONG']
    RESP_SHORT = event_id['RESP/SHORT']

    codes = events[:, 2]
    samples = events[:, 0]

    # STIM sequence and mapping: original trial index over STIMs only
    is_stim = np.isin(codes, [STIM_LONG, STIM_SHORT])
    stim_samples = samples[is_stim]
    n_trials = len(stim_samples)
    trial_idx = np.arange(n_trials, dtype=int)
    sample_to_trial = {int(s): int(i) for s, i in zip(stim_samples, trial_idx)}

    # Pair STIM?RESP within each condition: RESP sample -> original STIM trial index
    def _pair(stim_samp, resp_samp):
        stim_samp = np.asarray(stim_samp)
        resp_samp = np.sort(np.asarray(resp_samp))
        out = {}
        j = 0
        for s in stim_samp:
            while j < len(resp_samp) and resp_samp[j] <= s:
                j += 1
            if j < len(resp_samp):
                out[int(resp_samp[j])] = int(sample_to_trial[int(s)])
                j += 1
        return out

    resp_map = {}
    resp_map.update(_pair(samples[codes == STIM_LONG], samples[codes == RESP_LONG]))
    resp_map.update(_pair(samples[codes == STIM_SHORT], samples[codes == RESP_SHORT]))
    # resp_map: RESP sample -> original STIM trial index

    # ---- Which epochs currently exist (everything, since no rejection yet) ----
    kept = epochs.events
    kept_samples = kept[:, 0]
    kept_codes = kept[:, 2]

    # Map which STIM/RESP epochs exist for each original trial
    stim_kept = np.zeros(n_trials, dtype=bool)
    resp_kept = np.zeros(n_trials, dtype=bool)
    for s, c in zip(kept_samples, kept_codes):
        s = int(s)
        if c in (STIM_LONG, STIM_SHORT):
            stim_kept[sample_to_trial[s]] = True
        elif c in (RESP_LONG, RESP_SHORT):
            t = resp_map.get(s, None)
            if t is not None:
                resp_kept[t] = True

    # Align behavior to existing STIM epochs (drop trials without a STIM epoch)
    behav_df = behav_df[stim_kept].reset_index(drop=True)

    # For those surviving STIM trials, carry whether a RESP epoch exists
    behav_df['stim_kept'] = True
    behav_df['resp_kept'] = resp_kept[stim_kept].astype(bool)

    # ---- Remove long-RT trials (RT > 1.5 s) from BOTH epochs and behavior ----
    rt_bad_pos = np.flatnonzero(behav_df['response_time'].to_numpy() > 1.5)   # positions in current behav_df
    if rt_bad_pos.size:
        print(f"[INFO] Removing {rt_bad_pos.size} trials with RT > 1.5 s.")

        # Map positions in behav_df back to ORIGINAL STIM trial indices
        # Original trial indices that survived the earlier "stim_kept" filter:
        orig_idx_kept = np.flatnonzero(stim_kept)                # original indices of kept STIM trials
        bad_orig_trials = set(orig_idx_kept[rt_bad_pos].tolist())

        # Build epoch-level keep mask
        ev = epochs.events
        eid = epochs.event_id
        is_stim_epoch = np.isin(ev[:, 2], [eid['STIM/LONG'], eid['STIM/SHORT']])
        is_resp_epoch = np.isin(ev[:, 2], [eid['RESP/LONG'], eid['RESP/SHORT']])

        keep_mask = np.ones(len(epochs), dtype=bool)

        # Drop STIM epochs whose original trial is in bad_orig_trials
        for i_ep, (samp, is_st) in enumerate(zip(ev[:, 0], is_stim_epoch)):
            if not is_st:
                continue
            trial = sample_to_trial.get(int(samp), None)
            if trial is not None and trial in bad_orig_trials:
                keep_mask[i_ep] = False

        # Drop RESP epochs whose paired original STIM trial is in bad_orig_trials
        for i_ep, (samp, is_rp) in enumerate(zip(ev[:, 0], is_resp_epoch)):
            if not is_rp:
                continue
            trial = resp_map.get(int(samp), None)  # original STIM trial index
            if trial is not None and trial in bad_orig_trials:
                keep_mask[i_ep] = False

        # Apply to epochs
        epochs = epochs[keep_mask]

        # Update behavior flags: mark those trials as not kept
        behav_df.loc[rt_bad_pos, 'stim_kept'] = False
        behav_df.loc[rt_bad_pos, 'resp_kept'] = False

        # And finally drop those rows from the saved behavioral table
        behav_df = behav_df.drop(index=rt_bad_pos).reset_index(drop=True)

    # ---- Save outputs ----
    epoch_path = op.join(sub_path, f"{subject_id}-epo.fif")
    epochs.save(epoch_path, fmt="double", overwrite=True, verbose=False)
    print("EPOCHS SAVED:", epoch_path)

    epoch_behav_path = op.join(sub_path, f"{subject_id}-epo-behav.csv")
    behav_df.to_csv(epoch_behav_path, index=False)
    print("EPO-BEHAV SAVED:", epoch_behav_path)


if __name__ == '__main__':
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
