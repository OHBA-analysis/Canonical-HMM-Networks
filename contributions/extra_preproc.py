"""Extra Preprocessing functions."""

from mne.preprocessing import ICA
from mne_icalabel import label_components
import matplotlib.pyplot as plt


def ica_ICLabel(raw, picks="eeg", n_components=30, method="iclabel", threshold=0.0):
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
    if method == "iclabel":
        print("Applying average reference (ICLabel requirement)...")
        raw_fit.set_eeg_reference("average", verbose=False)

    # 2. Fit ICA (Extended Infomax)
    print(
        f"Fitting ICA (method='infomax', n_components={n_components}, picks={picks})..."
    )
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
    probs = ic_labels["y_pred_proba"]  # Get predicted probabilities

    # 4. Identify bad components (combining labels and confidence threshold)
    exclude_idx = []
    print(f"Filtering artifact components (threshold > {threshold:.2f})...")

    for idx, (label, prob) in enumerate(zip(labels, probs)):
        # Exclude "brain" and "other"
        if label not in ["brain", "other"]:
            # Only mark for removal if confidence is above the set threshold
            if prob > threshold:
                exclude_idx.append(idx)
                print(
                    f"  ICA{idx:03d}: {label} (confidence: {prob:.2f}) -> [Marked for removal]"
                )
            else:
                print(
                    f"  ICA{idx:03d}: {label} (confidence: {prob:.2f}) -> [Kept] (Confidence below threshold)"
                )
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
                ax.set_title(new_title, fontsize=10, fontweight="bold")

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
