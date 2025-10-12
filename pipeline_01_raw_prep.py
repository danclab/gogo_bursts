import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

import mne
from mne import pick_channels_regexp
from mne.io.ctf.trans import _make_ctf_coord_trans_set
from mne.transforms import apply_trans
from mne.annotations import events_from_annotations

import pandas as pd

from utilities import files

def pair_stim_resp_strict(stim_idx, stim_samples, resp_samples, cond_name):
    """
    Pair each STIM with the first subsequent RESP.
    Enforces 1:1 matching and raises ValueError if any mismatch occurs.
    Returns (stim_idx_aligned, stim_samp_aligned, resp_samp_aligned).
    """
    stim_idx = np.asarray(stim_idx)
    stim_samples = np.asarray(stim_samples)
    resp_samples = np.asarray(resp_samples)

    # Basic count check
    if len(stim_samples) != len(resp_samples):
        raise ValueError(
            f"[{cond_name}] Count mismatch: {len(stim_samples)} STIM vs {len(resp_samples)} RESP."
        )

    # Ensure RESP are in ascending time
    resp_samples = np.sort(resp_samples, kind='mergesort')

    aligned_idx, aligned_stim, aligned_resp = [], [], []
    j = 0
    n_resp = len(resp_samples)

    for idx, s in zip(stim_idx, stim_samples):
        # advance until first RESP strictly after this STIM
        while j < n_resp and resp_samples[j] <= s:
            j += 1
        if j >= n_resp:
            raise ValueError(
                f"[{cond_name}] No RESP found after STIM at sample {int(s)} (trial_idx={int(idx)})."
            )
        aligned_idx.append(idx)
        aligned_stim.append(s)
        aligned_resp.append(resp_samples[j])
        j += 1

    # Final bijection check
    if len(aligned_idx) != len(stim_samples):
        raise ValueError(
            f"[{cond_name}] Pairing failure: paired {len(aligned_idx)} of {len(stim_samples)} trials."
        )

    return np.asarray(aligned_idx), np.asarray(aligned_stim), np.asarray(aligned_resp)


def run(subject_id, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)
    path = parameters["dataset_path"]
    sfreq = parameters["downsample_dataset"]
    data_path = op.join(path, "data_v2")
    der_path = op.join(path, "derivatives_v2")
    files.make_folder(der_path)

    proc_path = op.join(der_path, "processed")
    files.make_folder(proc_path)

    raw_fname = op.join(data_path, f'{subject_id}_GOGO-raw.fif')
    if not op.exists(raw_fname):
        return None

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)
    if not op.exists(sub_path):
        return None

    files.make_folder(sub_path)

    qc_folder = op.join(sub_path, "QC")
    files.make_folder(qc_folder)


    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.rename_channels(lambda x: x.replace('-3907', '').replace('-3908', ''))

    raw_events, event_id = events_from_annotations(raw)

    # Event codes
    RESP_LONG = event_id['RESP/LONG']
    RESP_SHORT = event_id['RESP/SHORT']
    STIM_LONG = event_id['STIM/LONG']
    STIM_SHORT = event_id['STIM/SHORT']

    # Identify consecutive responses and drop the 2nd (and further) in each run
    codes = raw_events[:, 2]
    is_resp = np.isin(codes, [RESP_LONG, RESP_SHORT])

    # current is RESP and immediately previous is RESP -> drop current
    prev_is_resp = np.r_[False, is_resp[:-1]]
    drop_mask = is_resp & prev_is_resp
    keep_mask = ~drop_mask

    if drop_mask.any():
        n_drop_total = int(drop_mask.sum())
        n_drop_long = int(((codes == RESP_LONG) & drop_mask).sum())
        n_drop_short = int(((codes == RESP_SHORT) & drop_mask).sum())
        print(f"[CLEAN] Removed {n_drop_total} consecutive RESP events "
              f"(LONG: {n_drop_long}, SHORT: {n_drop_short}).")

    raw_events = raw_events[keep_mask, :]

    # Recompute helpers after cleaning
    sfreq = float(raw.info['sfreq'])
    codes = raw_events[:, 2]
    samples = raw_events[:, 0]
    rows = np.arange(len(raw_events))

    # Trial counter over STIMs only (0-based; add +1 if you prefer 1-based)
    is_stim = np.isin(codes, [STIM_LONG, STIM_SHORT])
    stim_rows = rows[is_stim]
    stim_order = -np.ones(len(raw_events), dtype=int)
    stim_order[stim_rows] = np.arange(len(stim_rows))

    # Split by condition
    stim_long_rows = rows[codes == STIM_LONG]
    stim_short_rows = rows[codes == STIM_SHORT]
    stim_long_samp = samples[codes == STIM_LONG]
    stim_short_samp = samples[codes == STIM_SHORT]
    resp_long_samp = samples[codes == RESP_LONG]
    resp_short_samp = samples[codes == RESP_SHORT]

    # Strict pairing
    iL, sL, rL = pair_stim_resp_strict(stim_long_rows, stim_long_samp, resp_long_samp, "LONG")
    iS, sS, rS = pair_stim_resp_strict(stim_short_rows, stim_short_samp, resp_short_samp, "SHORT")

    # Use the STIM order as trial_idx (add +1 here if you want 1-based)
    df_long = pd.DataFrame({
        'subject_id': subject_id,
        'trial_idx': stim_order[iL],
        'condition': 'LONG',
        'response_time': (rL - sL) / sfreq
    })
    df_short = pd.DataFrame({
        'subject_id': subject_id,
        'trial_idx': stim_order[iS],
        'condition': 'SHORT',
        'response_time': (rS - sS) / sfreq
    })

    behav = pd.concat([df_long, df_short], ignore_index=True)
    behav = behav.sort_values('trial_idx').reset_index(drop=True)
    behav = behav[['subject_id', 'trial_idx', 'condition', 'response_time']]

    out_csv = op.join(sub_path, f'{subject_id}-behav.csv')
    behav.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Find time of last relevant event
    last_sample = raw_events[-1, 0]
    crop_tmax = (last_sample / raw.info['sfreq']) + 5.0

    # Crop raw data
    if crop_tmax<raw.times[-1]:
        raw.crop(tmax=crop_tmax)

    # Pick channels corresponding to the cHPI positions
    hpi_picks = pick_channels_regexp(raw.info['ch_names'], 'HLC00[123][123].*')

    # make sure we get 9 channels
    if len(hpi_picks) != 9:
        raise RuntimeError('Could not find all 9 cHPI channels')

    # get indices in alphabetical order
    sorted_picks = np.array(sorted(hpi_picks,
                                   key=lambda k: raw.info['ch_names'][k]))

    # make picks to match order of dig cardinial ident codes.
    # LPA (HPIC002[123]-*), NAS(HPIC001[123]-*), RPA(HPIC003[123]-*)
    hpi_picks = sorted_picks[[3, 4, 5, 0, 1, 2, 6, 7, 8]]
    del sorted_picks

    # process the entire run
    time_sl = slice(0, len(raw.times))
    #chpi_data = raw[hpi_picks, time_sl][0]
    chpi_data = raw.get_data()[hpi_picks, time_sl]

    # transforms
    tmp_trans = _make_ctf_coord_trans_set(None, None)
    ctf_dev_dev_t = tmp_trans['t_ctf_dev_dev']
    del tmp_trans

    # find indices where chpi locations change (threshold is 0.00001)
    indices = [0]
    indices.extend(np.where(np.any(np.abs(np.diff(chpi_data, axis=1))>0.00001,axis=0))[0]+ 1)
    # data in channels are in ctf device coordinates (cm)
    rrs = chpi_data[:, indices].T.reshape(len(indices), 3, 3)  # m
    # map to mne device coords
    rrs = apply_trans(ctf_dev_dev_t, rrs)
    gofs = np.ones(rrs.shape[:2])  # not encoded, set all good
    moments = np.zeros(rrs.shape)  # not encoded, set all zero
    times = raw.times[indices] + raw._first_time
    chpi_locs = dict(rrs=rrs, gofs=gofs, times=times, moments=moments)

    head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=False)

    used_coils = np.array([0, 1, 2])
    coil_labels = ['lpa', 'nas', 'rpa']

    plt.figure()

    plt.subplot(3, 1, 1)
    for idx, i in enumerate(used_coils):
        c = chpi_locs['rrs'][:, i, 0] - np.mean(chpi_locs['rrs'][:, i, 0])
        plt.plot(chpi_locs['times'], c * 1000, label=coil_labels[idx])
    plt.legend()
    # plt.ylim([-10,10])
    plt.xlim(chpi_locs['times'][[0, -1]])
    plt.ylabel('x (mm)')

    plt.subplot(3, 1, 2)
    for idx, i in enumerate(used_coils):
        c = chpi_locs['rrs'][:, i, 1] - np.mean(chpi_locs['rrs'][:, i, 1])
        plt.plot(chpi_locs['times'], c * 1000)
    # plt.ylim([-15,15])
    plt.xlim(chpi_locs['times'][[0, -1]])
    plt.ylabel('y (mm)')

    plt.subplot(3, 1, 3)
    for idx, i in enumerate(used_coils):
        c = chpi_locs['rrs'][:, i, 2] - np.mean(chpi_locs['rrs'][:, i, 2])
        plt.plot(chpi_locs['times'], c * 1000)
    # plt.ylim([-15,15])
    plt.xlim(chpi_locs['times'][[0, -1]])
    plt.ylabel('z (mm)')
    plt.xlabel('time (s)')

    plt.savefig(
        op.join(qc_folder, "{}-chpi.png".format(subject_id)),
        dpi=150, bbox_inches="tight"
    )
    plt.close("all")

    for idx, i in enumerate(used_coils):
        sd = np.std(chpi_locs['rrs'][:, i, 0]) * 1000
        print(f'{coil_labels[idx]}, x SD={sd:.2f} mm')
    for idx, i in enumerate(used_coils):
        sd = np.std(chpi_locs['rrs'][:, i, 1]) * 1000
        print(f'{coil_labels[idx]}, y SD={sd:.2f} mm')
    for idx, i in enumerate(used_coils):
        sd = np.std(chpi_locs['rrs'][:, i, 2]) * 1000
        print(f'{coil_labels[idx]}, z SD={sd:.2f} mm')

    lpa_pos = chpi_locs['rrs'][:, used_coils[0], :]
    nas_pos = chpi_locs['rrs'][:, used_coils[1], :]
    rpa_pos = chpi_locs['rrs'][:, used_coils[2], :]

    lpa_rpa_dist = np.sqrt(np.sum((lpa_pos - rpa_pos) ** 2, axis=-1))
    lpa_nas_dist = np.sqrt(np.sum((lpa_pos - nas_pos) ** 2, axis=-1))
    rpa_nas_dist = np.sqrt(np.sum((rpa_pos - nas_pos) ** 2, axis=-1))

    plt.figure()
    plt.plot(lpa_rpa_dist, label='lpa-rpa')
    plt.plot(lpa_nas_dist, label='lpa-nas')
    plt.plot(rpa_nas_dist, label='rpa-nas')
    plt.legend()
    plt.savefig(
        op.join(qc_folder, "{}-chpi_dists.png".format(subject_id)),
        dpi=150, bbox_inches="tight"
    )
    plt.close("all")

    print(f'LPA-RPA = {np.mean(lpa_rpa_dist) * 1000} mm')
    print(f'LPA-NAS = {np.mean(lpa_nas_dist) * 1000} mm')
    print(f'RPA-NAS = {np.mean(rpa_nas_dist) * 1000} mm')

    fig = mne.viz.plot_head_positions(head_pos, mode="traces", show=False)
    fig.savefig(
        op.join(qc_folder, "{}-head_pos.png".format(subject_id)),
        dpi=150, bbox_inches="tight"
    )

    fig = mne.viz.plot_head_positions(
        head_pos, mode="field", destination=raw.info["dev_head_t"], info=raw.info,
        show=False
    )  # visualization 3D
    fig.savefig(
        op.join(qc_folder, "{}-head_pos_3d.png".format(subject_id)),
        dpi=150, bbox_inches="tight"
    )

    raw_sss = mne.preprocessing.maxwell_filter(
        raw, head_pos=head_pos,
        st_duration=10,
        origin=[0., 0., 0.04],
        coord_frame='head',
        verbose=True
    )

    raw_path = op.join(
        sub_path,
        "{}-raw.fif".format(subject_id)
    )
    eve_path = op.join(
        sub_path,
        "{}-eve.fif".format(subject_id)
    )

    raw_sss, events = raw_sss.copy().resample(
        sfreq,
        npad="auto",
        events=raw_events,
        n_jobs=-1,
    )
    print(f'Duration after downsampling: {raw_sss.times[-1]}')

    raw_sss.save(
        raw_path,
        fmt="single",
        overwrite=True
    )

    print("RAW SAVED:", raw_path)

    raw_sss.close()

    mne.write_events(
        eve_path,
        events,
        overwrite=True
    )

    print("EVENTS SAVED:", eve_path)

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