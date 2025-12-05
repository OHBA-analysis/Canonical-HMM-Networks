"""Preprocessing functions."""

import mne
import numpy as np
from scipy import stats


def detect_bad_segments(
    raw,
    picks,
    mode=None,
    metric="std",
    window_length=None,
    significance_level=0.05,
    maximum_fraction=0.1,
    ref_meg="auto",
):
    """Bad segment detection using the G-ESD algorithm.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    picks : str or list of str
        Channel type to pick.
    mode : str, optional
        None or 'diff' to take the difference fo the time series
        before detecting bad segments.
    metric : str, optional
        Either 'std' (for standard deivation) or 'kurtosis'.
    window_length : int, optional
        Window length to used to calculate statistics.
        Defaults to twice the sampling frequency.
    significance_level : float, optional
        Significance level (p-value) to consider as an outlier.
    maximum_fraction : float, optional
        Maximum fraction of time series to mark as bad.
    ref_meg : str, optional
        ref_meg argument to pass to mne.pick_types.

    Returns
    -------
    bad : np.ndarray
        Times of True (bad) or False (good) to indicate whether
        a time point is good or bad. This is the full length of
        the original time series. Shape is (n_samples,).
    """
    print()
    print("Bad segment detection")
    print("---------------------")

    if metric not in ["std", "kurtosis"]:
        raise ValueError("metric must be 'std' or 'kurtosis'.")

    if metric == "kurtosis":

        def _kurtosis(inputs):
            return stats.kurtosis(inputs, axis=None)

        metric_func = _kurtosis
    else:
        metric_func = np.std

    if window_length is None:
        window_length = int(raw.info["sfreq"] * 2)

    # Pick channels
    if picks == "eeg":
        chs = mne.pick_types(raw.info, eeg=True, exclude="bads")
    else:
        chs = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude="bads")

    # Get data
    data, times = raw.get_data(
        picks=chs, reject_by_annotation="omit", return_times=True
    )
    if mode == "diff":
        data = np.diff(data, axis=1)
        times = times[1:]

    # Calculate metric for each window
    metrics = []
    indices = []
    starts = np.arange(0, data.shape[1], window_length)
    for i in range(len(starts)):
        start = starts[i]
        if i == len(starts) - 1:
            stop = None
        else:
            stop = starts[i] + window_length
        m = metric_func(data[:, start:stop])
        metrics.append(m)
        indices += [i] * data[:, start:stop].shape[1]

    # Detect outliers
    bad_metrics_mask = _gesd(metrics, alpha=significance_level, p_out=maximum_fraction)
    bad_metrics_indices = np.where(bad_metrics_mask)[0]

    # Look up what indices in the original data are bad
    bad = np.isin(indices, bad_metrics_indices)

    # Make lists containing the start and end (index) of end bad segment
    onsets = np.where(np.diff(bad.astype(float)) == 1)[0] + 1
    if bad[0]:
        onsets = np.r_[0, onsets]
    offsets = np.where(np.diff(bad.astype(float)) == -1)[0] + 1
    if bad[-1]:
        offsets = np.r_[offsets, len(bad) - 1]
    assert len(onsets) == len(offsets)

    # Timing of the bad segments in seconds
    onsets = raw.first_samp / raw.info["sfreq"] + times[onsets.astype(int)]
    offsets = raw.first_samp / raw.info["sfreq"] + times[offsets.astype(int)]
    durations = offsets - onsets

    # Description for the annotation of the Raw object
    descriptions = np.repeat(f"bad_segment_{picks}", len(onsets))

    # Annotate the Raw object
    raw.annotations.append(onsets, durations, descriptions)

    # Summary statistics
    n_bad_segments = len(onsets)
    total_bad_time = durations.sum()
    total_time = raw.n_times / raw.info["sfreq"]
    percentage_bad = (total_bad_time / total_time) * 100

    # Print useful summary information
    print(f"Modality: {picks}")
    print(f"Mode: {mode}")
    print(f"Metric: {metric}")
    print(f"Significance level: {significance_level}")
    print(f"Maximum fraction: {maximum_fraction}")
    print(
        f"Found {n_bad_segments} bad segments: "
        f"{total_bad_time:.1f}/{total_time:.1f} "
        f"seconds rejected ({percentage_bad:.1f}%)"
    )

    return raw


def detect_bad_channels(
    raw,
    picks,
    fmin=2,
    fmax=80,
    n_fft=2000,
    significance_level=0.05,
    ref_meg="auto",
):
    """Detect bad channels using PSD and G-ESD outlier detection.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw data object.
    picks : str or list of str
        Channel types to pick.
    fmin, fmax : float
        Frequency range for PSD computation.
    n_fft : int
        FFT length for PSD.
    significance_level : float
        Significance level for GESD outlier detection.
    ref_meg : str, optional
        ref_meg argument to pass to mne.pick_types.

    Returns
    -------
    bad_ch_names : list of str
        Detected bad channel names.
    """
    print()
    print("Bad channel detection")
    print("---------------------")

    # Pick channels
    if picks == "eeg":
        chs = mne.pick_types(raw.info, eeg=True, exclude="bads")
    else:
        chs = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude="bads")

    # Compute PSD (bad channels excluded by MNE)
    psd = raw.compute_psd(
        picks=chs,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        reject_by_annotation=True,
        verbose=False,
    )
    pow_data = psd.get_data()

    if len(chs) != pow_data.shape[0]:
        raise RuntimeError(
            f"Channel mismatch: {len(chs)} chans vs PSD shape {pow_data.shape[0]}"
        )

    # Check for NaN or zero PSD
    bad_forced = [
        ch
        for ch, psd_ch in zip(chs, pow_data)
        if np.any(np.isnan(psd_ch)) or np.all(psd_ch == 0)
    ]
    if bad_forced:
        raise RuntimeError(
            f"PSD contains NaNs or all-zero values for channels: {bad_forced}"
        )

    # Metric for detecting outliers in
    pow_log = np.log10(pow_data)
    X = np.std(pow_log, axis=-1)

    # Detect artefacts with GESD
    mask = _gesd(X, alpha=significance_level)

    # Get the names for the bad channels
    chs = np.array(raw.ch_names)[chs]
    bads = list(chs[mask])

    # Mark bad channels in the Raw object
    raw.info["bads"] = bads

    print(f"{len(bads)} bad channels:")
    print(bads)

    return raw


def _gesd(X, alpha, p_out=1, outlier_side=0):
    """Detect outliers using Generalized ESD test.

    Parameters
    ----------
    X : list or np.ndarray
        data to detect outliers within. Must be a 1D array containing
        the metric we want to detect outliers for. E.g. a list of
        standard deviation for each window into a time series.
    alpha : float
        Significance level threshold for outliers.
    p_out : float
        Maximum fraction of time series to set as outliers.
    outlier_side : int, optional
        Can be{-1,0,1} :
        - -1 -> outliers are all smaller
        -  0 -> outliers could be small/negative or large/positive
        -  1 -> outliers are all larger

    Returns
    -------
    mask : np.ndarray
        Boolean mask for bad segments. Same shape as X.

    Notes
    -----
    B. Rosner (1983). Percentage Points for a Generalized ESD
    Many-Outlier Procedure. Technometrics 25(2), pp. 165-172.
    """
    if outlier_side == 0:
        alpha = alpha / 2
    n_out = int(np.ceil(len(X) * p_out))
    if np.any(np.isnan(X)):
        y = np.where(np.isnan(X))[0]
        idx1, x2 = _gesd(X[np.isfinite(X)], alpha, n_out, outlier_side)
        idx = np.zeros_like(X).astype(bool)
        idx[y[idx1]] = True
    n = len(X)
    temp = X.copy()
    R = np.zeros(n_out)
    rm_idx = np.zeros(n_out, dtype=int)
    lam = np.zeros(n_out)
    for j in range(0, int(n_out)):
        i = j + 1
        if outlier_side == -1:
            rm_idx[j] = np.nanargmin(temp)
            sample = np.nanmin(temp)
            R[j] = np.nanmean(temp) - sample
        elif outlier_side == 0:
            rm_idx[j] = int(np.nanargmax(abs(temp - np.nanmean(temp))))
            R[j] = np.nanmax(abs(temp - np.nanmean(temp)))
        elif outlier_side == 1:
            rm_idx[j] = np.nanargmax(temp)
            sample = np.nanmax(temp)
            R[j] = sample - np.nanmean(temp)
        R[j] = R[j] / np.nanstd(temp)
        temp[int(rm_idx[j])] = np.nan
        p = 1 - alpha / (n - i + 1)
        t = stats.t.ppf(p, n - i - 1)
        lam[j] = ((n - i) * t) / (np.sqrt((n - i - 1 + t**2) * (n - i + 1)))
    mask = np.zeros(n).astype(bool)
    mask[rm_idx[np.where(R > lam)[0]]] = True
    return mask
