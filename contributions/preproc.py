"""Preprocessing functions."""

import mne
import numpy as np
from scipy import stats
from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt
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


def ica_ICLabel(
    raw, 
    picks='eeg', 
    n_components=30, 
    method='iclabel', 
    threshold=0.0
):
    """
    Automatic artifact removal using ICA and automatic labeling algorithms (ICLabel or MEGNet).
    website: https://mne.tools/mne-icalabel/stable/index.html
    paper: https://doi.org/10.1016/j.neuroimage.2019.05.026
    Supports EEG (ICLabel) and MEG (MEGNet) data, and allows setting a confidence threshold.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    picks : str or list of str
        Channel types to pick for ICA.
        - For EEG data, typically 'eeg'.
        - For MEG data, can be 'meg', 'mag', 'grad', etc.
    n_components : int, optional
        Number of PCA components to use for ICA decomposition. Defaults to 15.
    method : str, optional
        Labeling method.
        - 'iclabel': For EEG data.
        - 'megnet': For MEG data.
        Defaults to 'iclabel'.
    threshold : float, optional
        Confidence threshold (0.0 to 1.0).
        Components are only removed if labeled as artifact (non-brain/other) 
        and the predicted probability is greater than this threshold. 
        Defaults to 0.0 (removes if labeled as artifact regardless of probability).

    Returns
    -------
    raw : mne.io.Raw
        The cleaned Raw object (modified in-place).
    """
    print()
    print(f"ICA Automatic Artifact Removal (Method: {method})")
    print("----------------------------")

    # 1. Create a copy for ICA fitting
    # High-pass filtering at 1Hz is generally recommended for best ICA decomposition results.
    # We do this on a copy to avoid changing the original data's filter settings.
    print("Filtering data copy (1-100Hz) for ICA fitting...")
    raw_fit = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)

    # If ICLabel (EEG), average reference must be applied
    if method == 'iclabel':
        print("Applying average reference (ICLabel requirement)...")
        raw_fit.set_eeg_reference("average", verbose=False)
    
    # 2. Fit ICA (Extended Infomax)
    print(f"Fitting ICA (method='infomax', n_components={n_components}, picks={picks})...")
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=97,
        verbose=False,
    )
    ica.fit(raw_fit, picks=picks)

    # 3. Label components using specified method
    print(f"Labeling components using {method}...")
    try:
        ic_labels = label_components(raw_fit, ica, method=method)
    except Exception as e:
        print(f"Labeling failed: {e}")
        return raw

    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"] # Get predicted probabilities

    # 4. Identify bad components (combining labels and confidence threshold)
    exclude_idx = []
    print(f"Filtering artifact components (threshold > {threshold:.2f})...")
    
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        # Exclude "brain" and "other"
        if label not in ["brain", "other"]:
            # Only mark for removal if confidence is above the set threshold
            if prob > threshold:
                exclude_idx.append(idx)
                print(f"  ICA{idx:03d}: {label} (confidence: {prob:.2f}) -> [Marked for removal]")
            else:
                print(f"  ICA{idx:03d}: {label} (confidence: {prob:.2f}) -> [Kept] (Confidence below threshold)")
        else:
            # If brain or other, can also print for reference
            # print(f"  ICA{idx:03d}: {label} (confidence: {prob:.2f}) -> [Kept]")
            pass

    # 5. Plot removed components (with labels and confidence)
    if len(exclude_idx) > 0:
        print(f"Found {len(exclude_idx)} artifact components to remove. Plotting...")
        
        # Use MNE's plot_components to generate figure
        title = f"Removed Artifacts ({method}, thresh={threshold})"
        fig = ica.plot_components(picks=exclude_idx, title=title, show=False)
        
        # Compatibility handling
        if isinstance(fig, list):
            fig = fig[0]
            
        # Modify subplot titles
        topo_axes = [ax for ax in fig.axes if ax.get_title().startswith("ICA")]
        
        # Ensure axis count matches to avoid indexing errors
        if len(topo_axes) == len(exclude_idx):
            for i, ax in enumerate(topo_axes):
                comp_idx = exclude_idx[i]
                label_name = labels[comp_idx]
                label_prob = probs[comp_idx]
                
                # Title format: ICA001 \n Eye blink (0.99)
                new_title = f"ICA{comp_idx:03d}\n{label_name} ({label_prob:.2f})"
                ax.set_title(new_title, fontsize=10, fontweight='bold')
        
        plt.show()
    else:
        print("No artifact components found meeting the criteria.")

    # 6. Apply cleaning to original data
    if len(exclude_idx) > 0:
        print("Applying ICA cleaning to original data...")
        ica.apply(raw, exclude=exclude_idx)
    else:
        print("No components removed, data remains unchanged.")

    return raw