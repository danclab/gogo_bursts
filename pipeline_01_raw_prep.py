import csv
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import os.path as op

from mne import pick_channels_regexp
from mne.io.ctf.trans import _make_ctf_coord_trans_set
from mne.transforms import apply_trans

from utilities import files
import mne
from mne.annotations import events_from_annotations

def run(group, subject_id, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)
    path = parameters["dataset_path"]
    sfreq = parameters["downsample_dataset"]
    data_path = op.join(path, "data", group)
    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)

    proc_path = op.join(der_path, "processed", group)
    files.make_folder(proc_path)

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)

    qc_folder = op.join(sub_path, "QC")
    files.make_folder(qc_folder)

    raw = mne.io.read_raw_fif(op.join(data_path, f'{subject_id}_GOGO-raw.fif'))
    raw.rename_channels(lambda x: x.replace('-3907', '').replace('-3908', ''))

    raw_events, event_id = events_from_annotations(raw)

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
    chpi_data = raw[hpi_picks, time_sl][0]

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
    # try:
    #     index = int(sys.argv[1])
    # except:
    #     print("incorrect arguments")
    #     sys.exit()
    #
    # try:
    #     json_file = sys.argv[2]
    #     print("USING:", json_file)
    # except:
    #     json_file = "settings.json"
    #     print("USING:", json_file)

    json_file = "settings.json"
    # run('ASD', 'COM013', json_file)
    # run('ASD', 'COM023', json_file)
    # run('TD', 'COM033', json_file)
    run('TD', 'COM040', json_file)