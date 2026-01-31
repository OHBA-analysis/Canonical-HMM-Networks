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
    raw : mne.io.Raw
        MNE Raw object.
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


def detect_bad_channels(*args, mode="std", **kwargs):
    """Detect bad channels.

    See _detect_bad_channels_std and _detect_bad_channels_psd for
    arguments that can be passed to this function.

    Parameters
    ----------
    mode : str, optional
        Either 'std' or 'psd'.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object with bad channels marked.
    """
    if mode == "std":
        return _detect_bad_channels_std(*args, **kwargs)
    elif mode == "psd":
        return _detect_bad_channels_psd(*args, **kwargs)
    else:
        raise ValueError(f"mode must be 'std' or 'psd', got '{mode}'.")


def _detect_bad_channels_std(raw, picks, ref_meg="auto", significance_level=0.05):
    """Detect bad channels using the G-ESD algorithm based on standard deviation.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE raw object.
    picks : str
        Channel types to pick. See Notes for recommendations.
    ref_meg : str
        ref_meg argument to pass with mne.pick_types.
    significance_level : float
        Significance level for detecting outliers. Must be between 0-1.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object with bad channels marked.

    Notes
    -----
    For Elekta/MEGIN data, we recommend using picks='mag' or picks='grad'
    separately (in no particular order).

    Note that with CTF data, mne.pick_types will return:
        ~274 axial grads (as magnetometers) if picks='mag', ref_meg=False
        ~28 reference axial grads if picks='grad'.
    Thus, it is recommended to use picks='mag' in combination with ref_mag=False,
    and picks='grad' separately (in no particular order).
    """
    print()
    print("Bad channel detection (std)")
    print("---------------------------")

    # Select channels
    if (picks == "mag") or (picks == "grad"):
        ch_inds = mne.pick_types(raw.info, meg=picks, ref_meg=ref_meg, exclude="bads")
    elif picks == "meg":
        ch_inds = mne.pick_types(raw.info, meg=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "eeg":
        ch_inds = mne.pick_types(raw.info, eeg=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "eog":
        ch_inds = mne.pick_types(raw.info, eog=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "ecg":
        ch_inds = mne.pick_types(raw.info, ecg=True, ref_meg=ref_meg, exclude="bads")
    elif picks == "misc":
        ch_inds = mne.pick_types(raw.info, misc=True, exclude="bads")
    else:
        raise NotImplementedError(f"picks={picks} not available.")
    ch_names = np.array(raw.ch_names)[ch_inds]

    # Calculate standard deviation for each channel
    data = raw.get_data(picks=ch_inds)
    std = np.std(data, axis=-1)

    # Detect outliers
    mask = _gesd(std, alpha=significance_level)
    chs = np.array(raw.ch_names)[ch_inds]
    bads = list(chs[mask])

    # Mark as bad
    for bad in bads:
        if bad not in raw.info["bads"]:
            raw.info["bads"] += bads

    # Print useful summary information
    print(f"{len(bads)} bad channels:")
    print(bads)

    return raw


def _detect_bad_channels_psd(
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
    raw : mne.io.Raw
        MNE Raw object with bad channels marked.
    """
    print()
    print("Bad channel detection (psd)")
    print("---------------------------")

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
    bads = chs[mask]

    # Mark bad channels in the Raw object
    for bad in bads:
        if bad not in raw.info["bads"]:
            raw.info["bads"] += bads

    # Print useful summary information
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


def decimate_headshape_points(
    raw,
    decimate_amount=0.01,  # Decimate head mesh (1 cm bins)
    include_facial_info=True,  # Keep facial points for ICP
    remove_zlim=-0.02,  # Remove >2 cm below nasion
    angle=0,  # No rotation
    method="gridaverage",  # Grid-based averaging
    face_Z=[-0.06, 0.02],  # Keep face: -6 to +2 cm (up-down)
    face_Y=[0.06, 0.15],  # 6–15 cm forward
    face_X=[-0.03, 0.03],  # ±3 cm left-right
    decimate_facial_info=True,
    decimate_facial_info_amount=0.01,  # Decimate face (1 cm bins)
):
    """Decimate headshape points.

    Useful for reducing the number of headshape points collected using an
    EinScan for OPM recordings.

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw object.
    decimate_amount : float, optional
        Bin width in metres to decimate.
    include_facial_info : bool, optional
        Should we keep facial headshape points?
    remove_zlim : float, optional
        Remove headshape points below this z-value (in metres).
    angle : float, optional
        How much should we rotate the headshape points?
    method : str, optional
        What method should we use for decimation?
    face_Z : list, optional
        Keep headshape points within these z-values (in metres).
    face_Y : list, optional
        Keep headshape points within these y-values (in metres).
    face_X : list, optional
        Keep headshape points within these x-values (in metres).
    decimate_facial_info : bool, optional
        Should we decimate facial headshape points?
    decimate_facial_info_amount : float, optional
        Bin width in metres to decimate.

    Returns
    -------
    raw : mne.io.Raw
        MNE Raw object.
    """
    print()
    print("Decimate headshape points")
    print("-------------------------")

    dig = raw.info["dig"]
    headshape = np.array([d["r"] for d in dig if "r" in d])
    print("Digitization points:", headshape.shape)

    decimated_headshape = _decimate_headshape(
        headshape,
        decimate_amount=decimate_amount,
        include_facial_info=include_facial_info,
        remove_zlim=remove_zlim,
        angle=angle,
        method=method,
        face_Z=face_Z,
        face_Y=face_Y,
        face_X=face_X,
        decimate_facial_info=decimate_facial_info,
        decimate_facial_info_amount=decimate_facial_info_amount,
    )

    # Initialize fiducial positions
    fid_positions = {"nasion": None, "lpa": None, "rpa": None}

    # Extract fiducials from the dig points
    for f in dig:
        if f["coord_frame"] == 4:  # Ensure head coordinate frame
            if f["ident"] == 2 and fid_positions["nasion"] is None:
                fid_positions["nasion"] = f["r"]
            elif f["ident"] == 1 and fid_positions["lpa"] is None:
                fid_positions["lpa"] = f["r"]
            elif f["ident"] == 3 and fid_positions["rpa"] is None:
                fid_positions["rpa"] = f["r"]

    # Verify the extracted fiducials
    if any(v is None for v in fid_positions.values()):
        raise RuntimeError(
            "One or more fiducials (nasion, LPA, RPA) not found in "
            "the head coordinate frame."
        )

    # Create a DigMontage using the extracted fiducials
    # and decimated headshape points
    montage = mne.channels.make_dig_montage(
        hsp=decimated_headshape,
        nasion=fid_positions["nasion"],
        lpa=fid_positions["lpa"],
        rpa=fid_positions["rpa"],
        coord_frame="head",
    )

    # Set the new montage
    return raw.set_montage(montage)


def _decimate_headshape(
    headshape,
    decimate_amount=0.015,  # average over 1.5cm
    include_facial_info=True,
    remove_zlim=0.02,  # Remove 2cm above nasion
    angle=10,  # At an angle of 10deg
    method="gridaverage",
    face_Z=[-0.08, 0.02],  # Z-axis (up-down) -8cm to 2cm
    face_Y=[0.06, 0.15],  # Y-axis (forward-back) 6 to 15cm
    face_X=[-0.07, 0.07],  # X-axis (left_right) -7 to 7cm
    decimate_facial_info=True,
    decimate_facial_info_amount=0.008,  # average over 0.8cm
):
    """Decimate headshape points.

    Parameters
    ----------
    - headshape : np.ndarray
        Nx3 array of headshape points in meters.
    - include_facial_info : bool
        Include facial points if True.
    - remove_zlim : float
        Remove points above nasion on the z-axis in meters.
    - method : str
        Downsampling method. Note: only method supported is 'gridaverage'.
    - facial_info_above_z (float): float
        Max z-value for facial points in meters.
    - facial_info_below_z : float
        Min z-value for facial points in meters.
    - facial_info_above_y : float
        Max y-value for facial points in meters.
    - facial_info_below_y : float
        Min y-value for facial points in meters.
    - facial_info_below_x : float
        Min x-value for facial points in meters.
    - decimate_facial_info : bool
        Whether to decimate facial points.
    - decimate_facial_info_amount : float
        Grid size for downsampling facial info in meters.

    Returns
    -------
    decimated_headshape : np.ndarray
        Decimated headshape points.
    """
    if include_facial_info:
        facial_mask = (
            (headshape[:, 2] > face_Z[0])
            & (headshape[:, 2] < face_Z[1])
            & (headshape[:, 1] > face_Y[0])
            & (headshape[:, 1] < face_Y[1])
            & (headshape[:, 0] > face_X[0])
            & (headshape[:, 0] < face_X[1])
        )
        facial_points = headshape[facial_mask]
        if decimate_facial_info:
            facial_points = _grid_average_decimate(
                facial_points, decimate_facial_info_amount
            )
    if remove_zlim is not None:
        print("Removing points below zlim")
        rotated_headshape = _rotate_pointcloud(headshape, angle, "x")
        z_mask = rotated_headshape[:, 2] > remove_zlim
        filtered_rotated_points = rotated_headshape[z_mask]
        headshape = _rotate_pointcloud(filtered_rotated_points, -angle, "x")
    if method == "gridaverage":
        print(f"Using {method}")
        headshape = _grid_average_decimate(headshape, decimate_amount)
    else:
        raise ValueError(f"Unsupported decimation method: {method}")
    if include_facial_info:
        headshape = np.vstack((headshape, facial_points))
    return headshape


def _rotate_pointcloud(points, angle_degrees, axis="x"):
    """
    Rotates the point cloud around a specified axis.

    Parameters
    ----------
    points : np.ndarray
        Headshape points
    angle_degrees : float
        Amount to rotate in degrees.
    axis : str
        Axis to rotate.
    """
    angle_radians = np.radians(angle_degrees)
    if axis == "x":
        rotation_matrix = np.array(
            [
                [1, 0, 0],
                [0, np.cos(angle_radians), -np.sin(angle_radians)],
                [0, np.sin(angle_radians), np.cos(angle_radians)],
            ]
        )
    elif axis == "y":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), 0, np.sin(angle_radians)],
                [0, 1, 0],
                [-np.sin(angle_radians), 0, np.cos(angle_radians)],
            ]
        )
    elif axis == "z":
        rotation_matrix = np.array(
            [
                [np.cos(angle_radians), -np.sin(angle_radians), 0],
                [np.sin(angle_radians), np.cos(angle_radians), 0],
                [0, 0, 1],
            ]
        )
    else:
        raise ValueError("Invalid axis. Choose from 'x', 'y', or 'z'.")
    return np.dot(points, rotation_matrix.T)


def _grid_average_decimate(point_cloud, voxel_size):
    """Decimate a point cloud using grid averaging.

    This function divides the space into a voxel grid, computes the average
    position of points within each voxel, and returns a decimated point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray
        A numpy array of shape (N, 3) representing the point cloud, where N
        is the number of points, and each point has (x, y, z) coordinates.

    voxel_size : float
        The size of the voxel grid. Points within a grid cell are averaged
        to compute the decimated point.

    Returns
    -------
    decimated_cloud : np.ndarray
        A numpy array of shape (M, 3) representing the decimated point cloud,
        where M is the number of voxels containing points.

    Notes
    -----
    - This method assumes the input point cloud is dense and unstructured.
    - For very large point clouds, consider optimizing memory usage.
    """
    voxel_indices = np.floor(point_cloud / voxel_size).astype(np.int32)
    voxel_dict = {}
    for idx, point in zip(voxel_indices, point_cloud):
        key = tuple(idx)
        if key not in voxel_dict:
            voxel_dict[key] = []
        voxel_dict[key].append(point)
    return np.array([np.mean(voxel_dict[key], axis=0) for key in voxel_dict])
