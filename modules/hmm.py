"""Hidden Markov Model (HMM) Utility Functions."""

import glob
import re
import os
from collections import defaultdict, OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from osl_dynamics import files
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.utils import plotting


def prepare_data_for_canonical_hmm(data, parcellation):
    """Prepare data for a canonical HMM.

    Parameters
    ----------
    data : osl_dynamics.data.Data
        An osl-dynamics Data object.
    parcellation : str
        Name of the parcellation.

    Returns
    -------
    data : osl_dynamics.data.Data
        An osl-dynamics Data object with the prepared data.
    """

    # Validation
    available_parcellations = ["38ROI_Giles", "52ROI_Glasser", "Elekta"]
    if parcellation not in available_parcellations:
        raise ValueError(f"parcellation much be one of: {available_parcellations}")

    # Prepare data for a sensor-level HMM
    if parcellation == "Elekta":
        pca_components_1 = np.load(f"models/{parcellation}/pca_components_1.npy")
        pca_components_2 = np.load(f"models/{parcellation}/pca_components_2.npy")
        data.prepare(
            {
                "pca": {"pca_components": pca_components_1},
                "tde_pca": {"n_embeddings": 15, "pca_components": pca_components_2},
                "standardize": {},
            }
        )

    # Prepare data for a parcel-level HMM
    else:
        pca_components = np.load(f"models/{parcellation}/pca_components.npy")
        template_cov = np.load(f"models/{parcellation}/template_cov.npy")
        data.prepare(
            {
                "align_channel_signs": {
                    "template_cov": template_cov,
                    "n_embeddings": 15,
                },
                "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
                "standardize": {},
            }
        )

    return data


def load_canonical_hmm(n_states, parcellation, sequence_length=400, batch_size=64):
    """Load a canonical HMM.

    Parameters
    ----------
    n_states : int
        Number of states.
    parcellation : str
        Name of the parcellation.
    sequence_length : int, optional
        Sequence length.
    batch_size : int, optional
        Batch size.

    Returns
    -------
    model : osl_dynamics.models.hmm.Model
        The canonical HMM.
    """

    # Validation
    available_parcellations = ["38ROI_Giles", "52ROI_Glasser", "Elekta"]
    if parcellation not in available_parcellations:
        raise ValueError(f"parcellation much be one of: {available_parcellations}")

    # Load HMM parameters
    means = np.load(f"models/{parcellation}/{n_states:02d}_states/means.npy")
    covs = np.load(f"models/{parcellation}/{n_states:02d}_states/covs.npy")
    trans_prob = np.load(f"models/{parcellation}/{n_states:02d}_states/trans_prob.npy")
    initial_state_probs = np.load(
        f"models/{parcellation}/{n_states:02d}_states/initial_state_probs.npy"
    )

    # Create a model
    config = Config(
        n_states=n_states,
        n_channels=means.shape[-1],
        sequence_length=sequence_length,
        learn_means=False,
        learn_covariances=True,
        initial_means=means,
        initial_covariances=covs,
        initial_trans_prob=trans_prob,
        initial_state_probs=initial_state_probs,
        batch_size=batch_size,
        learning_rate=0.01,  # we won't train the model, this hyperparameter doesn't matter
        n_epochs=20,  # we won't train the model, this hyperparameter doesn't matter
    )
    model = Model(config)

    return model


def plot_canonical_group_level_networks(n_states, parcellation, plots_dir="plots"):
    """Plot networks for a particular canonical HMM.

    Parameters
    ----------
    n_states : int
        Number of states.
    parcellation : str
        Name of the parcellation.
    plots_dir : str, optional
        Directory to save png files to.
    """
    os.makedirs(plots_dir, exist_ok=True)

    if parcellation == "38ROI_Giles":
        parcellation_file = "atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz"
    elif parcellation == "52ROI_Glasser":
        parcellation_file = "atlas-Glasser_nparc-52_space-MNI_res-8x8x8.nii.gz"
    elif parcellation == "Elekta":
        parcellation == None
    else:
        raise ValueError(f"{parcellation} unavailable.")

    # Load data
    model_dir = f"models/{parcellation}/{n_states:02d}_states"
    f = np.load(f"{model_dir}/f.npy")
    psds = np.load(f"{model_dir}/psds.npy")
    pow_maps = np.load(f"{model_dir}/pow_maps.npy")

    # PSDs
    if n_states > 10:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("tab10")
    plotting.set_style(
        {
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 28,
            "lines.linewidth": 4,
        }
    )
    p = np.mean(psds, axis=1)  # average over parcels
    p0 = np.mean(psds, axis=(0, 1))  # average over states and parcels
    for i in range(p.shape[0]):
        fig, ax = plotting.plot_line(
            [f],
            [p0],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
            x_range=[f[0], f[-1]],
            y_range=[0, 0.15],
            plot_kwargs={"color": "black", "linestyle": "--"},
        )
        ax.plot(f, p[i], color=cmap(i))
        plotting.save(fig, f"{plots_dir}/psd_{i:02d}.png", tight_layout=True)

    if parcellation == "Elekta":
        # Power maps
        plotting.set_style(
            {
                "font.size": 40,
                "xtick.labelsize": 40,
                "ytick.labelsize": 40,
            }
        )
        for i in range(pow_maps.shape[0]):
            plotting.topoplot(
                layout="neuromag306mag",
                data=pow_maps[i],
                channel_names=np.load(
                    files.scanner.path / "neuromag306_channel_names.npy"
                ),
                plot_boxes=False,
                show_deleted_sensors=True,
                show_names=False,
                colorbar=True,
                cmap="cold_hot",
                n_contours=25,
                filename=f"{plots_dir}/pow_{i:02d}.png",
            )

    else:
        # Power maps
        plotting.set_style(
            {
                "axes.labelsize": 20,
                "xtick.labelsize": 20,
                "ytick.labelsize": 20,
            }
        )
        power.save(
            pow_maps,
            mask_file="MNI152_T1_8mm_brain.nii.gz",
            parcellation_file=parcellation_file,
            subtract_mean=True,
            filename=f"{plots_dir}/pow_.png",
        )

        # Coherence networks
        plotting.set_style(
            {
                "axes.labelsize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            }
        )
        coh_nets = np.load(f"{model_dir}/coh_nets.npy")
        coh_nets -= np.average(coh_nets, axis=0)
        coh_nets = connectivity.threshold(coh_nets, percentile=98, absolute_value=True)
        connectivity.save(
            coh_nets,
            parcellation_file=parcellation_file,
            plot_kwargs={"display_mode": "xz", "annotate": False},
            filename=f"{plots_dir}/coh_.png",
        )


def display_network_plots(n_states, plots_dir):
    """Find and display canonical network plots in Jupyter Notebook.

    Parameters
    ----------
    n_states : int
        Number of states.
    plots_dir : str
        Path to directory where the plots were saved.
    """
    modalities_expected = ["coh", "pow", "psd"]  # order of columns in the display
    image_glob = f"{plots_dir}/*.png"
    figsize_per_row = (3 * 3, 3)  # width,height per row (tweak as desired)

    # find files
    files = sorted(glob.glob(image_glob))

    # regex to capture prefix and number like "psd_02.png", "coh-3.png", "pow3.png"
    # captures letters in group1 and digits in group2
    pattern = re.compile(r"([A-Za-z]+)[-_]?0*(\d+)\.png$", flags=re.IGNORECASE)

    # mapping: state_index -> { modality -> filepath, ... }
    by_state = defaultdict(dict)
    unknown_files = []

    for f in files:
        fn = os.path.basename(f)
        m = pattern.search(fn)
        if not m:
            unknown_files.append(fn)
            continue
        modality = m.group(1).lower()
        num = int(m.group(2))
        # prefer direct mapping if within range, else fallback to modulo
        if num < n_states:
            state = num
        else:
            state = num % n_states
            # optional: you could log this to inspect unexpected indices
            print(
                f"Warning: index {num} in {fn} >= n_states ({n_states}); "
                f"using {state} (num % n_states)."
            )
        by_state[state][modality] = f

    # optionally show any files that couldn't be parsed
    if unknown_files:
        print("Could not parse these filenames (skipped):", unknown_files)

    # Display grid: one row per state (0..n_states-1),
    # columns in modalities_expected order
    for state in range(n_states):
        row_map = by_state.get(state, {})
        n_cols = len(modalities_expected)
        fig, axes = plt.subplots(
            1, n_cols, figsize=(figsize_per_row[0], figsize_per_row[1])
        )
        if n_cols == 1:
            axes = [axes]
        for col_idx, mod in enumerate(modalities_expected):
            ax = axes[col_idx]
            fp = row_map.get(mod)
            if fp and os.path.exists(fp):
                img = mpimg.imread(fp)
                ax.imshow(img)
                ax.set_title(f"State {state}\n{os.path.basename(fp)}", fontsize=10)
                ax.axis("off")
            else:
                # empty placeholder for missing image
                ax.text(
                    0.5, 0.5, f"Missing\n{mod}", ha="center", va="center", fontsize=12
                )
                ax.set_title(f"State {state}\n{mod}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        plt.tight_layout()
        plt.show()
