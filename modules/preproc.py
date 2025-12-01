"""Preprocessing functions."""

import mne
import numpy as np
from scipy import stats
from mne.preprocessing import ICA
from mne_icalabel import label_components

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


import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components

def ica_ICLabel(
    raw, 
    picks='eeg', 
    n_components=15, 
    method='iclabel', 
    threshold=0.0
):
    """
    使用 ICA 和自动标记算法 (ICLabel 或 MEGNet) 进行自动伪影去除。
    
    支持 EEG (ICLabel) 和 MEG (MEGNet) 数据，并允许设置置信度阈值。

    Parameters
    ----------
    raw : mne.io.Raw
        MNE Raw 对象。
    picks : str or list of str
        用于 ICA 的通道类型。
        - 对于 EEG 数据，通常使用 'eeg'。
        - 对于 MEG 数据，可以使用 'meg', 'mag', 'grad' 等。
    n_components : int, optional
        用于 ICA 分解的 PCA 成分数量。默认为 15。
    method : str, optional
        标记方法。
        - 'iclabel': 适用于 EEG 数据。
        - 'megnet': 适用于 MEG 数据。
        默认为 'iclabel'。
    threshold : float, optional
        置信度阈值 (0.0 到 1.0)。
        只有当成分被标记为伪影 (非 brain/other) 且预测概率大于此阈值时，
        才会被去除。默认为 0.0 (即只要标记为伪影就去除)。

    Returns
    -------
    raw : mne.io.Raw
        清洗后的 Raw 对象（原地修改）。
    """
    print()
    print(f"ICA 自动伪影去除 (Method: {method})")
    print("----------------------------")

    # 1. 创建用于 ICA 拟合的副本
    # 为了获得最佳的 ICA 分解效果，通常建议进行 1Hz 高通滤波。
    # 我们在一个副本上进行此操作，以免改变原始数据的滤波设置。
    print("正在过滤数据副本 (1-100Hz) 以进行 ICA 拟合...")
    raw_fit = raw.copy().filter(l_freq=1.0, h_freq=100.0, verbose=False)

    # 如果是 ICLabel (EEG)，必须应用平均参考
    if method == 'iclabel':
        print("应用平均参考 (ICLabel 要求)...")
        raw_fit.set_eeg_reference("average", verbose=False)
    
    # 2. 拟合 ICA (扩展 Infomax)
    print(f"正在拟合 ICA (method='infomax', n_components={n_components}, picks={picks})...")
    ica = ICA(
        n_components=n_components,
        method="infomax",
        fit_params=dict(extended=True),
        random_state=97,
        verbose=False,
    )
    ica.fit(raw_fit, picks=picks)

    # 3. 使用指定方法标记成分
    print(f"正在使用 {method} 标记成分...")
    try:
        ic_labels = label_components(raw_fit, ica, method=method)
    except Exception as e:
        print(f"标记失败: {e}")
        return raw

    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"] # 获取预测概率

    # 4. 识别坏成分 (结合标签和置信度阈值)
    exclude_idx = []
    print(f"正在筛选伪影成分 (阈值 > {threshold:.2f})...")
    
    for idx, (label, prob) in enumerate(zip(labels, probs)):
        # 排除 "brain" (脑电) 和 "other" (其他/未知)
        if label not in ["brain", "other"]:
            # 只有当置信度高于设定的阈值时才标记为去除
            if prob > threshold:
                exclude_idx.append(idx)
                print(f"  ICA{idx:03d}: {label} (置信度: {prob:.2f}) -> [标记去除]")
            else:
                print(f"  ICA{idx:03d}: {label} (置信度: {prob:.2f}) -> [保留] (置信度低于阈值)")
        else:
            # 如果是 brain 或 other，也可以打印一下以供参考
            # print(f"  ICA{idx:03d}: {label} (置信度: {prob:.2f}) -> [保留]")
            pass

    # 5. 绘制被去除的成分 (带标签和置信度)
    if len(exclude_idx) > 0:
        print(f"总计发现 {len(exclude_idx)} 个待去除的伪影成分。正在绘图...")
        
        # 使用 MNE 的 plot_components 生成图像
        title = f"Removed Artifacts ({method}, thresh={threshold})"
        fig = ica.plot_components(picks=exclude_idx, title=title, show=False)
        
        # 兼容性处理
        if isinstance(fig, list):
            fig = fig[0]
            
        # 修改子图标题
        topo_axes = [ax for ax in fig.axes if ax.get_title().startswith("ICA")]
        
        # 确保轴的数量匹配，避免索引错误
        if len(topo_axes) == len(exclude_idx):
            for i, ax in enumerate(topo_axes):
                comp_idx = exclude_idx[i]
                label_name = labels[comp_idx]
                label_prob = probs[comp_idx]
                
                # 标题格式: ICA001 \n Eye blink (0.99)
                new_title = f"ICA{comp_idx:03d}\n{label_name} ({label_prob:.2f})"
                ax.set_title(new_title, fontsize=10, fontweight='bold')
        
        plt.show()
    else:
        print("未发现满足条件的伪影成分。")

    # 6. 将清洗应用到原始数据
    if len(exclude_idx) > 0:
        print("正在将 ICA 清洗应用到原始数据...")
        ica.apply(raw, exclude=exclude_idx)
    else:
        print("没有成分被去除，数据保持原样。")

    return raw