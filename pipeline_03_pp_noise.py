import sys
import json
import mne
import os.path as op

from extra.tools import dump_the_dict
from utilities import files

def run(group, subject_id, json_file):
    # opening a json file
    with open(json_file) as pipeline_file:
        parameters = json.load(pipeline_file)

    path = parameters["dataset_path"]
    der_path = op.join(path, "derivatives")
    files.make_folder(der_path)
    proc_path = op.join(der_path, "processed", group)
    files.make_folder(proc_path)

    print("ID:", subject_id)

    sub_path = op.join(proc_path, subject_id)
    files.make_folder(sub_path)

    qc_folder = op.join(sub_path, "QC")
    files.make_folder(qc_folder)

    raw_paths = files.get_files(sub_path, "zapline-" + subject_id, "-raw.fif")[2]
    raw_paths.sort()
    event_paths = files.get_files(sub_path, subject_id, "-eve.fif")[2]
    event_paths.sort()

    raw_eve_list = list(zip(raw_paths, event_paths))

    ica_json = dict()

    for raw_path, eve_path in raw_eve_list:
        print("INPUT RAW FILE:", raw_path)
        print("EVE_RAW MATCH:", raw_path.split("-")[-2] == eve_path.split("-")[-2])

        raw = mne.io.read_raw_fif(raw_path, verbose=False, preload=False)
        events = mne.read_events(eve_path)

        raw_filtered = raw.copy()
        raw_filtered = raw_filtered.pick_types(meg=True, eeg=False, ref_meg=False)
        raw_filtered.load_data().crop(
            tmin=raw_filtered.times[events[0, 0]],
            tmax=raw_filtered.times[events[-1, 0]]
        )
        raw_filtered.filter(
            l_freq=1.,
            h_freq=60,
            n_jobs=-1
        )

        ica = mne.preprocessing.ICA(
            method="infomax",
            fit_params=dict(extended=True),
            n_components=25,
            max_iter=5000
        )
        ica.fit(raw_filtered)

        ica_name = "{}-ica.fif".format(subject_id)

        ica_file = op.join(
            sub_path,
            ica_name
        )

        ica.save(ica_file, overwrite=True)

        ica_json[ica_name] = []

    ica_json_path = op.join(
        sub_path,
        "{}-ICA_to_reject.json".format(subject_id)
    )
    if not op.exists(ica_json_path):
        dump_the_dict(
            ica_json_path,
            ica_json
        )


if __name__=='__main__':
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