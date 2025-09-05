import copy
import json
import sys

from meegkit.dss import dss_line
from scipy.signal import welch
from mne.time_frequency import psd_array_multitaper

from utilities import files
import os.path as op
import mne
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

def nan_basic_interp(array):
    nans, ix = np.isnan(array), lambda x: x.nonzero()[0]
    array[nans] = np.interp(ix(nans), ix(~nans), array[~nans])
    return array


def zapline_until_gone(data, target_freq, sfreq, win_sz=10, spot_sz=2.5, viz=False, prefix="zapline_iter",
                       max_iter=100):
    """
    Returns: clean data, number of iterations

    Function iteratively applies the Zapline algorithm.

    data: assumed that the function is a part of the MNE-Python pipeline,
    the input should be an output of {MNE object}.get_data() function. The shape
    should be Trials x Sensors x Time for epochs.
    target_freq: frequency + harmonics that comb-like approach will be applied
    with Zapline
    sfreq: sampling frequency, the output of {MNE object}.info["sfreq"]
    win_sz: 2x win_sz = window around the target frequency
    spot_sz: 2x spot_sz = width of the frequency peak to remove
    viz: produce a visual output of each iteration,
    prefix: provide a path and first part of the file
    "{prefix}_{iteration number}.png"
    """

    iterations = 0
    aggr_resid = []

    freq_rn = [target_freq - win_sz, target_freq + win_sz]
    freq_sp = [target_freq - spot_sz, target_freq + spot_sz]

    norm_vals = []
    resid_lims = []
    real_target=None

    while True:
        if iterations > 0:
            if iterations >= max_iter:
                break
            if real_target is None:
                real_target=freq[freq_rn_ix[0]:freq_rn_ix[1]][np.argmax(mean_psd)]
                print(f'Real target={real_target}')
            data, art = dss_line(data.transpose(), real_target, sfreq, nremove=1)
            del art
            data = data.transpose()
        psd, freq = psd_array_multitaper(data, sfreq, verbose=False, n_jobs=30)

        freq_rn_ix = [
            np.where(freq >= freq_rn[0])[0][0],
            np.where(freq <= freq_rn[1])[0][-1]
        ]
        freq_used = freq[freq_rn_ix[0]:freq_rn_ix[1]]
        freq_sp_ix = [
            np.where(freq_used >= freq_sp[0])[0][0],
            np.where(freq_used <= freq_sp[1])[0][-1]
        ]

        norm_psd = psd[:, freq_rn_ix[0]:freq_rn_ix[1]]
        for ch_idx in range(norm_psd.shape[0]):
            if iterations == 0:
                norm_val = np.max(norm_psd[ch_idx, :])
                norm_vals.append(norm_val)
            else:
                norm_val = norm_vals[ch_idx]
            norm_psd[ch_idx, :] = norm_psd[ch_idx, :] / norm_val
        mean_psd = np.mean(norm_psd, axis=0)

        mean_psd_wospot = copy.copy(mean_psd)
        mean_psd_wospot[freq_sp_ix[0]: freq_sp_ix[1]] = np.nan
        mean_psd_tf = nan_basic_interp(mean_psd_wospot)
        pf = np.polyfit(freq_used, mean_psd_tf, 3)
        p = np.poly1d(pf)
        clean_fit_line = p(freq_used)
        residuals = mean_psd - clean_fit_line
        aggr_resid.append(np.mean(residuals))
        tf_ix = np.where(freq_used <= target_freq)[0][-1]
        print("Iteration:", iterations, "Power above the fit:", residuals[tf_ix])

        if viz:
            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 6), facecolor="gray", gridspec_kw={"wspace": 0.2})
            for sensor in range(psd.shape[0]):
                ax1.plot(freq_used, norm_psd[sensor, :])
            ax1.set_title("Normalized mean PSD \nacross trials")

            ax2.plot(freq_used, mean_psd_tf, c="gray")
            ax2.plot(freq_used, mean_psd, c="blue")
            ax2.plot(freq_used, clean_fit_line, c="red")
            ax2.set_title("Mean PSD across \ntrials and sensors")

            ax3.set_title("Residuals")
            tf_ix = np.where(freq_used <= target_freq)[0][-1]
            ax3.plot(residuals, freq_used)
            scat_color = "green"
            if residuals[tf_ix] <= 0:
                scat_color = "red"
            ax3.scatter(residuals[tf_ix], freq_used[tf_ix], c=scat_color)
            if iterations == 0:
                resid_lims = ax3.get_xlim()
            else:
                ax3.set_xlim(resid_lims)

            ax4.set_title("Iterations")

            ax4.scatter(np.arange(iterations + 1), aggr_resid)
            plt.savefig("{}_{}.png".format(prefix, str(iterations).zfill(3)))
            plt.close("all")

        if iterations > 0 and residuals[tf_ix] <= 0:
            break

        iterations += 1

    return [data, iterations]



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

    raw_paths = files.get_files(sub_path, subject_id, "-raw.fif")[2]
    raw_paths.sort()
    event_paths = files.get_files(sub_path, subject_id, "-eve.fif")[2]
    event_paths.sort()

    raw_eve_list = list(zip(raw_paths, event_paths))

    for raw_path, eve_path in raw_eve_list:
        print("INPUT RAW FILE:", raw_path)
        print("EVE_RAW MATCH:", raw_path.split("-")[-2] == eve_path.split("-")[-2])

        raw = mne.io.read_raw_fif(raw_path, verbose=False)
        raw = raw.pick_types(meg=True, eeg=True, ref_meg=True)
        fig = raw.plot_psd(
            tmax=np.inf, fmax=260, average=True, show=False, picks="meg"
        )
        fig.suptitle(subject_id)
        plt.savefig(
            op.join(qc_folder, "{}-raw-psd.png".format(subject_id)),
            dpi=150, bbox_inches="tight"
        )
        plt.close("all")

        info = raw.info
        raw_data = raw.get_data(picks='meg')

        zapped, iterations = zapline_until_gone(
            raw_data,
            60.0,
            info['sfreq'],
            win_sz=5,
            spot_sz=2,
            viz=True,
            prefix="{}/{}-50_1_iter".format(qc_folder, subject_id),
            max_iter=10
        )

        raw = mne.io.RawArray(
            np.vstack([zapped, raw.get_data(picks='eeg')]),
            info
        )
        fig = raw.plot_psd(tmax=np.inf, fmax=260, average=True, show=False, picks='meg')
        fig.suptitle(subject_id)
        plt.savefig(
            op.join(qc_folder, "{}-zapline-raw-psd.png".format(subject_id)),
            dpi=150,
            bbox_inches="tight"
        )
        plt.close("all")

        out_path = op.join(
            sub_path,
            "zapline-{}-raw.fif".format(subject_id)
        )

        raw.save(
            out_path,
            overwrite=True
        )
        del raw


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