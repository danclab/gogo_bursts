import sys
import json
import mne
import os.path as op
import numpy as np
from mne import events_from_annotations

from utilities import files


def run(group, subject_id, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    high_pass = parameters["high_pass_filter"]
    low_pass = parameters["low_pass_filter"]

    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed", group)
    files.make_folder(proc_path)

    print("ID:", subject_id)

    orig_raw = mne.io.read_raw_fif(op.join(path, 'data', group, f'{subject_id}_GOGO-raw.fif'))
    _, event_id = events_from_annotations(orig_raw)

    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)

    qc_folder = op.join(sub_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
    raw_paths.sort()

    ica_json_file = op.join(
        sub_path,
        "{}-ICA_to_reject.json".format(subject_id)
    )

    with open(ica_json_file) as ica_file:
        ica_files = json.load(ica_file)

    ica_keys = list(ica_files.keys())
    ica_keys.sort()

    event_paths = files.get_files(sub_path, subject_id, "-eve.fif")[2]
    event_paths.sort()

    raw_ica_eve = list(zip(raw_paths, ica_keys, event_paths))

    for (raw_path, ica_key, eve_path) in raw_ica_eve:
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

        epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=-2, tmax=2, baseline=None,
                            event_repeated='merge', preload=True)
        epoch_path = op.join(
            sub_path,
            "{}-epo.fif".format(subject_id)
        )

        epochs.save(
            epoch_path,
            fmt="double",
            overwrite=True,
            verbose=False,
        )


if __name__ == '__main__':
    # parsing command line arguments
    # try:
    #     subj_index = int(sys.argv[1])
    # except:
    #     print("incorrect arguments")
    #     sys.exit()
    #
    # try:
    #     sess_index = int(sys.argv[2])
    # except:
    #     print("incorrect arguments")
    #     sys.exit()
    #
    # try:
    #     json_file = sys.argv[3]
    #     print("USING:", json_file)
    # except:
    #     json_file = "settings.json"
    #     print("USING:", json_file)

    json_file = "settings.json"
    # run('ASD', 'COM013', json_file)
    # run('ASD', 'COM023', json_file)
    # run('TD', 'COM033', json_file)
    run('TD', 'COM040', json_file)