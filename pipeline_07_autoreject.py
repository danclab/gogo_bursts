import sys
import json
import os.path as op
from os import sep
import numpy as np
from mne import read_epochs, set_log_level
from utilities import files
from autoreject import AutoReject
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

set_log_level(verbose=False)

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

    epo_paths = files.get_files(sub_path, subject_id, "-epo.fif")[2]

    epo_paths.sort()

    for epo in epo_paths:
        epochs = read_epochs(epo, verbose=False, preload=True)
        print("AMOUNT OF EPOCHS:", len(epochs))

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

        cleaned = op.join(sub_path, "autoreject-" + epo.split(sep)[-1])
        epochs_ar.save(
            cleaned,
            overwrite=True
        )
        print("CLEANED EPOCHS SAVED:", cleaned)


if __name__=='__main__':
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