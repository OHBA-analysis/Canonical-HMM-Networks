"""Plot networks.

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from osl_dynamics.analysis import power, connectivity
from osl_dynamics.utils import plotting

def plot(n_states, parcellation, plots_dir="plots"):
    os.makedirs(plots_dir, exist_ok=True)

    if parcellation == "38ROI_Giles":
        model_dir = f"models/{parcellation}/{n_states:02d}_states"
        parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    elif parcellation == "52ROI_Glasser":
        model_dir = f"models/{parcellation}/{n_states:02d}_states"
        parcellation_file =  "Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz"
    else:
        raise ValueError(f"{parcellation} unavailable.")

    f = np.load(f"{model_dir}/f.npy")
    psds = np.load(f"{model_dir}/psds.npy")
    pow_maps = np.load(f"{model_dir}/pow_maps.npy")
    coh_nets = np.load(f"{model_dir}/coh_nets.npy")

    if n_states > 10:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("tab10")

    # PSDs
    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 28,
        "lines.linewidth": 4,
    })

    p = np.mean(psds, axis=1)  # average over parcels
    p0 = np.mean(psds, axis=(0, 1)) # average over states and parcels

    for i in range(p.shape[0]):
        fig, ax = plotting.plot_line(
            [f],
            [p0],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
            x_range=[f[0], f[-1]],
            y_range=[0, 0.105],
            plot_kwargs={"color": "black", "linestyle": "--"},
        )
        ax.plot(f, p[i], color=cmap(i))
        plotting.save(fig, f"plots/psd_{i:{len(str(n_states))}d}.png", tight_layout=True)

    # Power maps
    plotting.set_style({
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    power.save(
        pow_maps,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file=parcellation_file,
        subtract_mean=True,
        filename="plots/pow_.png",
    )

    # Coherence networks
    plotting.set_style({
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    coh_nets -= np.average(coh_nets, axis=0)
    coh_nets = connectivity.threshold(coh_nets, percentile=98, absolute_value=True)

    connectivity.save(
        coh_nets,
        parcellation_file=parcellation_file,
        plot_kwargs={"display_mode": "xz", "annotate": False},
        filename="plots/coh_.png",
    )

plot(n_states=8, parcellation="52ROI_Glasser")
