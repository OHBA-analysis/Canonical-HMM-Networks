"""Parcellation."""

import os
import mne
import scipy
import numpy as np
import nibabel as nib
from fsl import wrappers as fsl_wrappers

import osl_dynamics as osld
from . import source_recon


def parcellate(
    fns,
    voxel_data,
    voxel_coords,
    method,
    parcellation_file,
    orthogonalisation=None,
):
    """Parcellate data.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    voxel_data : np.ndarray
        (nvoxels x n_time) or (nvoxels x n_time x n_trials) and is assumed to be
        on the same grid as parcellation.
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation.
    method : str, optional
        'pca'           - take 1st PC of voxels
        'spatial_basis' - The parcel time-course for each spatial map is the
                          1st PC from all voxels, weighted by the spatial map.
        If the parcellation is unweighted and non-overlapping, 'spatial_basis'
        will give the same result as 'PCA' except with a different normalisation.
    parcellation_file : str
        Path to parcellation file. In same space as voxel_coords.
    orthogonalisation : str, optional
        Method for orthogonalising the data. Can be None or 'symmetric'.

    Returns
    -------
    parcel_data : np.ndarray
        Parcellated data. Shape is (parcels, time) or (parcels, time, epochs).
    """
    print("")
    print("Parcellating data")
    print("-----------------")

    if orthogonalisation not in [None, "symmetric"]:
        raise ValueError("orthogonalisation must be None or 'symmetric'.")

    if method not in ["pca", "spatial_basis"]:
        raise ValueError("method must be 'pca' or 'spatial_basis'.")

    # Get parcellation file
    parcellation_file = osld.files.check_exists(
        parcellation_file, osld.files.parcellation.directory
    )

    # Resample parcellation to match the mask
    parcellation = _resample_parcellation(fns, parcellation_file, voxel_coords)

    # Calculate parcel time courses
    parcel_data, _, _ = _get_parcel_data(voxel_data, parcellation, method=method)

    # Orthogonalisation
    if orthogonalisation == "symmetric":
        parcel_data = _symmetric_orthogonalisation(
            parcel_data, maintain_magnitudes=True
        )

    return parcel_data


def save_as_fif(parcel_data, raw, filename, extra_chans=None):
    """Save parcellated data as a fif file.

    Parameters
    ----------
    parcel_data : np.ndarray
        (parcels, time) or (parcels, time, epochs) data.
    raw : mne.Raw or mne.Epochs
        MNE Raw or Epochs objects to get info from.
    filename : str
        Output file path.
    extra_chans : str or list of str
        Extra channels, e.g. 'stim' or 'emg', to include in the parc_raw object.
        Defaults to 'stim'. stim channels are always added to parc_raw if they
        are present in raw.
    """
    print(f"Saving {filename}")

    if isinstance(raw, mne.Epochs):
        # Save as a MNE Epochs object
        parc_epo = _convert2mne_epochs(parcel_data, raw)
        parc_epo.save(filename, overwrite=True)

    else:
        # Save as a MNE Raw object
        parc_raw = _convert2mne_raw(parcel_data, raw, extra_chans=extra_chans)
        parc_raw.save(filename, overwrite=True)


def plot_psds(parc_fif, parcellation_file, fmin=0.5, fmax=45, filename=None):
    """Plot PSD of each parcel time course.

    Parameters
    ----------
    parc_fif : mne.Raw or mne.Epochs
        MNE Raw or Epochs object containing the parcel data.
    parcellation_file : str
        Path to parcellation file.
    fmin : float, optional
        Minimum frequency.
    fmax : float, optional
        Maximum frequency.
    filename : str, optional
        Output filename.
    """
    if "epo.fif" in parc_fif:
        raw = mne.Epochs(parc_fif)
    else:
        raw = mne.io.read_raw_fif(parc_fif)

    fs = raw.info["sfreq"]
    parc_ts = raw.get_data(picks="misc", reject_by_annotation="omit")

    if parc_ts.ndim == 3:
        # Calculate PSD for each epoch individually and average
        psd = []
        for i in range(parc_ts.shape[-1]):
            f, p = scipy.signal.welch(parc_ts[..., i], fs=fs, nperseg=fs, nfft=fs * 2)
            psd.append(p)
        psd = np.mean(psd, axis=0)
    else:
        # Calcualte PSD of continuous data
        f, psd = scipy.signal.welch(parc_ts, fs=fs, nperseg=fs, nfft=fs * 2)

    # Plot
    osld.utils.plotting.plot_psd_topo(
        f,
        psd,
        parcellation_file=parcellation_file,
        frequency_range=[fmin, fmax],
        filename=filename,
    )


def _resample_parcellation(fns, parcellation_file, voxel_coords):
    """Resample parcellation.

    Resample the parcellation so that the voxel coords correspond (using nearest
    neighbour) to the passed in coords. Passed in voxel_coords and parcellation
    must be in the same space, e.g. MNI.

    Used to make sure that the parcellation's voxel coords are the same as the
    voxel coords for some time series data.

    Parameters
    ----------
    parcellation_file : str
        Path to parcellation file. In same space as voxel_coords.
    voxel_coords :
        (nvoxels x 3) coordinates in mm in same space as parcellation.

    Returns
    -------
    parcellation_asmatrix : np.ndarray
        (nvoxels x n_parcels) resampled parcellation
    """
    gridstep = source_recon._get_gridstep(voxel_coords.T / 1000)
    print(f"gridstep = {gridstep} mm")

    path, name = os.path.split(
        os.path.splitext(os.path.splitext(parcellation_file)[0])[0]
    )

    parcellation_resampled = f"{fns.src_dir}/{name}_{gridstep}mm.nii.gz"

    # Create standard brain of the required resolution
    #
    # Command: flirt -in <parcellation_file> -ref <parcellation_file> \
    #          -out <parcellation_resampled> -applyisoxfm <gridstep>
    #
    # Note, this call raises:
    #
    #   Warning: An input intended to be a single 3D volume has multiple
    #   timepoints. Input will be truncated to first volume, but this
    #   functionality is deprecated and will be removed in a future release.
    #
    # However, it doesn't look like the input be being truncated, the
    # resampled parcellation appears to be a 4D volume.
    fsl_wrappers.flirt(
        parcellation_file,
        parcellation_file,
        out=parcellation_resampled,
        applyisoxfm=gridstep,
    )
    print(f"Resampled parcellation: {parcellation_resampled}")

    n_parcels = nib.load(parcellation_resampled).get_fdata().shape[3]
    n_voxels = voxel_coords.shape[1]

    # parcellation_asmatrix will be the parcels mapped onto the same dipole
    # grid as voxel_coords
    print("Finding nearest neighbour voxel")
    parcellation_asmatrix = np.zeros([n_voxels, n_parcels])
    for i in range(n_parcels):
        coords, vals = source_recon._niimask2mmpointcloud(parcellation_resampled, i)
        kdtree = scipy.spatial.KDTree(coords.T)

        # Find each voxel_coords best matching coords and assign
        # the corresponding parcel value to
        for j in range(n_voxels):
            distance, index = kdtree.query(voxel_coords[:, j])

            # Exclude from parcel any voxel_coords that are further than
            # gridstep away from the best matching coords
            if distance < gridstep:
                parcellation_asmatrix[j, i] = vals[index]

    return parcellation_asmatrix


def _get_parcel_data(voxel_data, parcellation_asmatrix, method="spatial_basis"):
    """Calculate parcel time courses.

    Parameters
    ----------
    voxel_data : np.ndarray
        (nvoxels x n_time) or (nvoxels x n_time x n_trials) and is assumed to be
        on the same grid as parcellation.
    parcellation_asmatrix: np.ndarray
        (nvoxels x n_parcels) and is assumed to be on the same grid as
        voxel_data.
    method : str, optional
        'pca'           - take 1st PC of voxels
        'spatial_basis' - The parcel time-course for each spatial map is the
                          1st PC from all voxels, weighted by the spatial map.
        If the parcellation is unweighted and non-overlapping, 'spatial_basis'
        will give the same result as 'PCA' except with a different normalisation.

    Returns
    -------
    parcel_data : np.ndarray
        n_parcels x n_time, or n_parcels x n_time x n_trials
    voxel_weightings : np.ndarray
        nvoxels x n_parcels
        Voxel weightings for each parcel to compute parcel_data from
        voxel_data
    voxel_assignments : bool np.ndarray
        nvoxels x n_parcels
        Boolean assignments indicating for each voxel the winner takes all
        parcel it belongs to
    """
    print(f"Calculating parcel time courses with {method}")

    if parcellation_asmatrix.shape[0] != voxel_data.shape[0]:
        Exception(
            f"Parcellation has {parcellation_asmatrix.shape[0]} voxels, "
            f"but data has {voxel_data.shape[0]}"
        )

    if len(voxel_data.shape) == 2:
        # Add dim for trials
        voxel_data = np.expand_dims(voxel_data, axis=2)
        added_dim = True
    else:
        added_dim = False

    n_parcels = parcellation_asmatrix.shape[1]
    n_time = voxel_data.shape[1]
    n_trials = voxel_data.shape[2]

    # Combine the trials and time dimensions together, we will
    # re-separate them after the parcel times eries are computed
    voxel_data_reshaped = np.reshape(
        voxel_data, (voxel_data.shape[0], n_time * n_trials)
    )
    parcel_data_reshaped = np.zeros((n_parcels, n_time * n_trials))

    voxel_weightings = np.zeros(parcellation_asmatrix.shape)

    if method == "spatial_basis":
        # estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(
            np.std(voxel_data_reshaped, axis=1), np.finfo(float).eps
        )

        for pp in range(n_parcels):
            # Scale group maps so all have a positive peak of height 1 in case
            # there is a very noisy outlier, choose the sign from the top 5%
            # of magnitudes
            thresh = np.percentile(np.abs(parcellation_asmatrix[:, pp]), 95)
            mapsign = np.sign(
                np.mean(
                    parcellation_asmatrix[parcellation_asmatrix[:, pp] > thresh, pp]
                )
            )
            scaled_parcellation = (
                mapsign
                * parcellation_asmatrix[:, pp]
                / np.max(np.abs(parcellation_asmatrix[:, pp]))
            )

            # Weight all voxels by the spatial map in question.
            # Apply the mask first then weight to reduce memory use
            weighted_ts = voxel_data_reshaped[scaled_parcellation > 0, :]
            weighted_ts = np.multiply(
                weighted_ts,
                np.reshape(scaled_parcellation[scaled_parcellation > 0], [-1, 1]),
            )
            weighted_ts = weighted_ts - np.reshape(
                np.mean(weighted_ts, axis=1), [-1, 1]
            )

            # Perform SVD and take scores of 1st PC as the node time-series
            #
            # U is nVoxels by nComponents - the basis transformation
            # S*V holds nComponents by time sets of PCA scores
            # - the time series data in the new basis
            d, U = scipy.sparse.linalg.eigs(weighted_ts @ weighted_ts.T, k=1)
            U = np.real(U)
            d = np.real(d)
            S = np.sqrt(np.abs(np.real(d)))
            V = weighted_ts.T @ U / S
            pca_scores = S @ V.T

            # 0.5 is a decent arbitrary threshold used in fslnets after
            # playing with various maps
            this_mask = scaled_parcellation[scaled_parcellation > 0] > 0.5

            if np.any(this_mask):  # the mask is non-zero
                # U is the basis by which voxels in the mask are weighted to
                # form the scores of the 1st PC
                relative_weighting = np.abs(U[this_mask]) / np.sum(np.abs(U[this_mask]))
                ts_sign = np.sign(np.mean(U[this_mask]))
                ts_scale = np.dot(
                    np.reshape(relative_weighting, [-1]),
                    temporal_std[scaled_parcellation > 0][this_mask],
                )

                node_ts = (
                    ts_sign
                    * (ts_scale / np.maximum(np.std(pca_scores), np.finfo(float).eps))
                    * pca_scores
                )

                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * (
                        np.reshape(U, [-1])
                        * scaled_parcellation[scaled_parcellation > 0].T
                    )
                )

            else:
                print(
                    f"WARNING: An empty parcel mask was found for parcel {pp} "
                    "when calculating its time-courses\n"
                    "The parcel will have a flat zero time-course.\n"
                    "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(n_time * n_trials)
                inds = np.where(scaled_parcellation > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_data_reshaped[pp, :] = node_ts

    elif method == "pca":
        print(
            "PCA assumes a binary parcellation.\n"
            "Parcellation will be binarised if it is not already "
            "(any voxels >0 are set to 1, otherwise voxels are set to 0), "
            "i.e. any weightings will be ignored.\n"
        )

        # Check that each voxel is only a member of one parcel
        if any(np.sum(parcellation_asmatrix, axis=1) > 1):
            print(
                "WARNING: Each voxel is meant to be a member of at most one "
                "parcel, when using the PCA method.\nResults may not be sensible"
            )

        # Estimate temporal-STD of data for normalisation
        temporal_std = np.maximum(
            np.std(voxel_data_reshaped, axis=1), np.finfo(float).eps
        )

        # Perform PCA on each parcel and select 1st PC scores to represent parcel
        for pp in range(n_parcels):
            if any(parcellation_asmatrix[:, pp]):  # non-zero
                parcel_data = voxel_data_reshaped[parcellation_asmatrix[:, pp] > 0, :]
                parcel_data = parcel_data - np.reshape(
                    np.mean(parcel_data, axis=1), [-1, 1]
                )

                # Perform svd and take scores of 1st PC as the node time-series
                #
                # U is nVoxels by nComponents - the basis transformation
                # S*V holds nComponents by time sets of PCA scores
                # - the time series data in the new basis
                d, U = scipy.sparse.linalg.eigs(parcel_data @ parcel_data.T, k=1)
                U = np.real(U)
                d = np.real(d)
                S = np.sqrt(np.abs(np.real(d)))
                V = parcel_data.T @ U / S
                pca_scores = S @ V.T

                # Restore sign and scaling of parcel time-series
                # U indicates the weight with which each voxel in the parcel
                # contributes to the 1st PC
                relative_weighting = np.abs(U) / np.sum(np.abs(U))
                ts_sign = np.sign(np.mean(U))
                ts_scale = np.dot(
                    np.reshape(relative_weighting, [-1]),
                    temporal_std[parcellation_asmatrix[:, pp] > 0],
                )

                node_ts = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                ) * pca_scores

                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = (
                    ts_sign
                    * ts_scale
                    / np.maximum(np.std(pca_scores), np.finfo(float).eps)
                    * np.reshape(U, [-1])
                )

            else:
                print(
                    f"WARNING: An empty parcel mask was found for parcel {pp} "
                    "when calculating its time-courses\n"
                    "The parcel will have a flat zero time-course.\n"
                    "Check this does not cause further problems with the analysis.\n"
                )

                node_ts = np.zeros(n_time * n_trials)
                inds = np.where(parcellation_asmatrix[:, pp] > 0)[0]
                voxel_weightings[inds, pp] = 0

            parcel_data_reshaped[pp, :] = node_ts

    else:
        Exception("Invalid method specified")

    # Re-separate the trials and time dimensions
    parcel_data = np.reshape(parcel_data_reshaped, (n_parcels, n_time, n_trials))
    if added_dim:
        parcel_data = np.squeeze(parcel_data, axis=2)

    # Compute voxel_assignments using winner takes all
    voxel_assignments = np.zeros(voxel_weightings.shape)
    for ivoxel in range(voxel_weightings.shape[0]):
        winning_parcel = np.argmax(voxel_weightings[ivoxel, :])
        voxel_assignments[ivoxel, winning_parcel] = 1

    return parcel_data, voxel_weightings, voxel_assignments


def _symmetric_orthogonalisation(
    timeseries, maintain_magnitudes=False, compute_weights=False
):
    """Symmetric orthogonalisation.

    Returns orthonormal matrix L which is closest to A, as measured by the
    Frobenius norm of (L-A). The orthogonal matrix is constructed from a
    singular value decomposition of A.

    If maintain_magnitudes is True, returns the orthogonal matrix L, whose
    columns have the same magnitude as the respective columns of A, and which
    is closest to A, as measured by the Frobenius norm of (L-A).

    Parameters
    ----------
    timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) data to orthoganlise.
        In the latter case, the ntpts and ntrials dimensions are concatenated.
    maintain_magnitudes : bool
    compute_weights : bool

    Returns
    -------
    ortho_timeseries : numpy.ndarray
        (nparcels x ntpts) or (nparcels x ntpts x ntrials) orthoganalised data
    weights : numpy.ndarray
        (optional output depending on compute_weights flag) weighting matrix
        such that, ortho_timeseries = timeseries * weights

    References
    ----------
    Colclough, G. L., Brookes, M., Smith, S. M. and Woolrich, M. W.,
    "A symmetric multivariate leakage correction for MEG connectomes,"
    NeuroImage 117, pp. 439-448 (2015)
    """
    print("Performing symmetric orthogonalisation")

    if len(timeseries.shape) == 2:
        # add dim for trials:
        timeseries = np.expand_dims(timeseries, axis=2)
        added_dim = True
    else:
        added_dim = False

    nparcels = timeseries.shape[0]
    ntpts = timeseries.shape[1]
    ntrials = timeseries.shape[2]
    compute_weights = False

    # combine the trials and time dimensions together,
    # we will re-separate them after the parcel timeseries are computed
    timeseries = np.transpose(np.reshape(timeseries, (nparcels, ntpts * ntrials)))

    if maintain_magnitudes:
        D = np.diag(np.sqrt(np.diag(np.transpose(timeseries) @ timeseries)))
        timeseries = timeseries @ D

    [U, S, V] = np.linalg.svd(timeseries, full_matrices=False)

    # we need to check that we have sufficient rank
    tol = max(timeseries.shape) * S[0] * np.finfo(type(timeseries[0, 0])).eps
    r = sum(S > tol)
    full_rank = r >= timeseries.shape[1]

    if full_rank:
        # polar factors of A
        ortho_timeseries = U @ np.conjugate(V)
    else:
        raise ValueError(
            "Not full rank, rank required is {}, but rank is only {}".format(
                timeseries.shape[1], r
            )
        )

    if compute_weights:
        # weights are a weighting matrix such that,
        # ortho_timeseries = timeseries * weights
        weights = np.transpose(V) @ np.diag(1.0 / S) @ np.conjugate(V)

    if maintain_magnitudes:
        # scale result
        ortho_timeseries = ortho_timeseries @ D

        if compute_weights:
            # weights are a weighting matrix such that,
            # ortho_timeseries = timeseries * weights
            weights = D @ weights @ D

    # Re-separate the trials and time dimensions
    ortho_timeseries = np.reshape(
        np.transpose(ortho_timeseries), (nparcels, ntpts, ntrials)
    )

    if added_dim:
        ortho_timeseries = np.squeeze(ortho_timeseries, axis=2)

    if compute_weights:
        return ortho_timeseries, weights
    else:
        return ortho_timeseries


def _convert2mne_raw(parc_data, raw, parcel_names=None, extra_chans="stim"):
    """Create and returns an MNE raw object that contains parcellated data.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels x ntpts) parcel data.
    raw : mne.Raw
        mne.io.raw object that produced parc_data via source recon and
        parcellation. Info such as timings and bad segments will be copied
        from this to parc_raw.
    parcel_names : list of str
        List of strings indicating names of parcels. If None then names are
        set to be parcel_0,...,parcel_{n_parcels-1}.
    extra_chans : str or list of str
        Extra channels, e.g. 'stim' or 'emg', to include in the parc_raw object.
        Defaults to 'stim'. stim channels are always added to parc_raw if they
        are present in raw.

    Returns
    -------
    parc_raw : mne.Raw
        Generated parcellation in mne.Raw format.
    """
    # What extra channels should we add to the parc_raw object?
    if extra_chans is None:
        extra_chans = []
    if isinstance(extra_chans, str):
        extra_chans = [extra_chans]
    extra_chans = np.unique(["stim"] + extra_chans)

    # parc_data is missing bad segments. For osl/rhino it's missing this data,
    # for mne solutions it's only missing the annotations (data shape is conserved)
    if (
        raw.get_data().shape[1] != parc_data.shape[1]
    ):  # We insert bad segments before creating the new MNE object
        _, times = raw.get_data(reject_by_annotation="omit", return_times=True)
        indices = raw.time_as_index(times, use_rounding=True)
        data = np.zeros([parc_data.shape[0], len(raw.times)], dtype=np.float32)
        data[:, indices] = parc_data
    else:
        data = parc_data

    # Create Info object
    info = raw.info
    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(data.shape[0])]
    parc_info = mne.create_info(
        ch_names=parcel_names, ch_types="misc", sfreq=info["sfreq"]
    )

    # Create Raw object
    parc_raw = mne.io.RawArray(data, parc_info)

    # Update filter info
    with parc_raw.info._unlock():
        parc_raw.info["highpass"] = float(raw.info["highpass"])
        parc_raw.info["lowpass"] = float(raw.info["lowpass"])

    # Copy timing info
    parc_raw.set_meas_date(raw.info["meas_date"])
    parc_raw.__dict__["_first_samps"] = raw.__dict__["_first_samps"]
    parc_raw.__dict__["_last_samps"] = raw.__dict__["_last_samps"]
    parc_raw.__dict__["_cropped_samp"] = raw.__dict__["_cropped_samp"]

    # Copy annotations from raw
    parc_raw.set_annotations(raw._annotations)

    # Add extra channels
    if "stim" not in raw:
        print("No stim channel to add to parc-raw.fif")
    for extra_chan in extra_chans:
        if extra_chan in raw:
            chan_raw = raw.copy().pick(extra_chan)
            chan_data = chan_raw.get_data()
            chan_info = mne.create_info(
                chan_raw.ch_names, raw.info["sfreq"], [extra_chan] * chan_data.shape[0]
            )
            chan_raw = mne.io.RawArray(chan_data, chan_info)
            parc_raw.add_channels([chan_raw], force_update_info=True)

    # Copy the description from the sensor-level Raw object
    parc_raw.info["description"] = raw.info["description"]

    return parc_raw


def _convert2mne_epochs(parc_data, epochs, parcel_names=None):
    """Create and returns an MNE Epochs object that contains parcellated data.

    Parameters
    ----------
    parc_data : np.ndarray
        (nparcels x ntpts x epochs) parcel data.
    epochs : mne.Epochs
        mne.io.raw object that produced parc_data via source recon and
        parcellation. Info such as timings and bad segments will be copied
        from this to parc_raw.
    parcel_names : list of str
        List of strings indicating names of parcels. If None then names are
        set to be parcel_0,...,parcel_{n_parcels-1}.

    Returns
    -------
    parc_epo : mne.Epochs
        Generated parcellation in mne.Epochs format.
    """

    # Epochs info
    info = epochs.info

    # Create parc info
    if parcel_names is None:
        parcel_names = [f"parcel_{i}" for i in range(parc_data.shape[0])]

    parc_info = mne.create_info(
        ch_names=parcel_names, ch_types="misc", sfreq=info["sfreq"]
    )
    parc_events = epochs.events

    # Parcellated data Epochs object
    parc_epo = mne.EpochsArray(np.swapaxes(parc_data.T, 1, 2), parc_info, parc_events)

    # Copy the description from the sensor-level Epochs object
    parc_epo.info["description"] = epochs.info["description"]

    return parc_epo
