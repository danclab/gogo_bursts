import json
import os.path as op
import pickle
import matplotlib.pyplot as plt
from fooof import FOOOF
from mne import read_epochs

from burst_detection import extract_bursts
from extra.tools import many_is_in
from superlet import scale_from_period, superlet
from utilities import files
import numpy as np


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

    epo_path = op.join(sub_path, f'autoreject-{subject_id}-epo.fif')
    epochs = read_epochs(epo_path, verbose=False, preload=True)

    epoch_types=['STIM','RESP']
    condition_names=['SHORT','LONG']

    all_bursts = {
        'subject_id': [],
        'group': [],
        'epoch_type': [],
        'condition': [],
        'channel': [],
        'trial': [],
        'waveform': np.zeros((0, 156)),
        'waveform_times': [],
        'peak_freq': [],
        'peak_amp_iter': [],
        'peak_amp_base': [],
        'peak_time': [],
        'peak_adjustment': [],
        'fwhm_freq': [],
        'fwhm_time': [],
        'polarity': [],
    }

    # Burst extraction and analysis
    for epo_type in epoch_types:
        for condition_name in condition_names:
            condition_epoch = epochs[f'{epo_type}/{condition_name}']

            sens = ["MLC1", "MLC25", "MLC32", "MLC42", "MLC54", "MLC55", "MLC63"]
            channels_used = [i for i in condition_epoch.info['ch_names'] if many_is_in(["MLC"], i)]
            channels_used = [i for i in channels_used if not many_is_in(sens, i)]

            for channel in channels_used:
                # Get data for the current sensor
                times = condition_epoch.times
                ch_idx = condition_epoch.ch_names.index(channel)
                ch_data = condition_epoch.get_data()[:, ch_idx, :]

                sfreq = condition_epoch.info['sfreq']
                max_freq = 120
                foi = np.linspace(.5, max_freq, 120)
                scales = scale_from_period(1 / foi)

                tf_trials = []

                # Compute time-frequency analysis
                for trial_idx in range(ch_data.shape[0]):
                    trial_tf = superlet(ch_data[trial_idx], sfreq, scales, 40, c_1=4, adaptive=True)
                    tf_trials.append(np.single(np.abs(trial_tf)))
                tf_trials = np.array(tf_trials)

                # Compute average power spectral density (PSD)
                average_psd = np.average(tf_trials, axis=(2, 0))

                # Fit the 1/f-like background
                ff = FOOOF()
                ff.fit(foi, average_psd, [.5, 120])
                ap = 10 ** ff._ap_fit  # Aperiodic component

                plt.figure()
                plt.plot(foi, average_psd, label='PSD')
                plt.plot(foi, ap, label='Aperiodic')
                plt.legend()
                plt.xlabel('Frequency (Hz)')
                plt.ylabel('Power')
                plt.title(f'{epo_type}-{condition_name}-{channel}')
                out_fname=op.join(qc_folder, f'{subject_id}_bursts_psd_{epo_type}_{condition_name}_{channel}.png')
                plt.savefig(out_fname)

                # Extract bursts
                search_range = np.where((foi >= 10) & (foi <= 33))[0]
                beta_lims = [13, 30]
                bursts = extract_bursts(
                    ch_data, tf_trials[:, search_range], times,
                    foi[search_range], beta_lims,
                    ap[search_range].reshape(-1, 1), sfreq
                )

                # Add metadata to bursts
                bursts['epoch_type'] = np.tile(epo_type, bursts['trial'].shape)
                bursts['condition'] = np.tile(condition_name, bursts['trial'].shape)
                bursts['channel'] = np.tile(channel, bursts['trial'].shape)
                bursts['subject_id'] = np.tile(subject_id, bursts['trial'].shape)
                bursts['group'] = np.tile(group, bursts['trial'].shape)

                plt.figure()
                plt.plot(bursts['waveform_times']*1000, np.mean(bursts['waveform'], axis=0)*1e15)
                plt.xlabel('Time (ms)')
                plt.ylabel('Amplitude (fT)')
                plt.title(f'{epo_type}-{condition_name}-{channel}: {len(bursts["trial"])} bursts')
                out_fname = op.join(qc_folder,
                                    f'{subject_id}_bursts_waveform_{epo_type}_{condition_name}_{channel}.png')
                plt.savefig(out_fname)

                # Append results
                for key in bursts.keys():
                    if key == 'waveform_times':
                        all_bursts[key] = bursts[key]
                    elif key == 'waveform':
                        all_bursts[key] = np.vstack([all_bursts[key], bursts[key]])
                    else:
                        all_bursts[key] = np.hstack([all_bursts[key], bursts[key]])

    output_file = op.join(sub_path, f'{subject_id}_bursts.pickle')
    with open(output_file, "wb") as fp:
        pickle.dump(all_bursts, fp)


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
    run('ASD', 'COM013', json_file)
    run('ASD', 'COM023', json_file)
    run('TD', 'COM033', json_file)
    run('TD', 'COM040', json_file)