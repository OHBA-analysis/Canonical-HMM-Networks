"""Source reconstruction."""

import os
import numpy as np
import nibabel as nib
from scipy.spatial import KDTree

import mne
from mne.beamformer._lcmv import _apply_lcmv
from mne.minimum_norm.inverse import _check_reference

from . import rhino, utils


def lcmv_beamformer(
    fns,
    data=None,
    chantypes=None,
    rank=None,
    frequency_range=None,
    pick_ori="max-power",
    reduce_rank=True,
    **kwargs,
):
    """Compute LCMV spatial filter.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    data : str | instance of mne.Raw | mne.Epochs, optional
        The measurement data to specify the channels to include. Bad channels
        in info['bads'] are not used. Will also be used to calculate data_cov.
        If None, fns.preproc_file is used.
    chantypes : list or str
        List of channel types to use to calculate the noise covariance.
        E.g. ['eeg'], ['mag', 'grad'], ['eeg', 'mag', 'grad'].
    rank : dict
        Calculate the rank only for a subset of channel types, and
        explicitly specify the rank for the remaining channel types.
        This can be extremely useful if you already know the rank of (part
        of) your data, for instance in case you have calculated it earlier.
        This parameter must be a dictionary whose keys correspond to channel
        types in the data (e.g. 'meg', 'mag', 'grad', 'eeg'), and whose
        values are integers representing the respective ranks. For example,
        {'mag': 90, 'eeg': 45} will assume a rank of 90 and 45 for
        magnetometer data and EEG data, respectively.
        The ranks for all channel types present in the data, but not
        specified in the dictionary will be estimated empirically. That is,
        if you passed a dataset containing magnetometer, gradiometer, and
        EEG data together with the dictionary from the previous example,
        only the gradiometer rank would be determined, while the specified
    frequency_range : list
        Lower and upper range (in Hz) for frequency range to bandpass filter.
        If None, no filtering is performed.
    pick_ori : None | 'normal' | 'max-power'
        The source orientation to compute the beamformer in.
    reduce_rank : bool
        Whether to reduce the rank by one during computation of the filter.
    **kwargs : keyword arguments
        Keyword arguments that will be passed to mne.beamformer.make_lcmv.

    Returns
    -------
    filters : instance of mne.beamformer.Beamformer
        Dictionary containing filter weights from LCMV beamformer.
        See: https://mne.tools/stable/generated/mne.beamformer.make_lcmv.html
    """
    print("")
    print("Making LCMV beamformer")
    print("----------------------")

    if data is None:
        data = fns.preproc_file

    if chantypes is None:
        raise ValueError("chantypes must be passed.")

    if isinstance(chantypes, str):
        chantypes = [chantypes]

    # Load data
    if isinstance(data, str):
        if "epo.fif" in data:
            data = mne.read_epochs(data)
        else:
            data = mne.io.read_raw_fif(data)

    # Bandpass filter
    if frequency_range is not None:
        data.filter(
            l_freq=frequency_range[0],
            h_freq=frequency_range[1],
            method="iir",
            iir_params={"order": 5, "ftype": "butter"},
        )

    # Load forward solution
    fwd = mne.read_forward_solution(fns.fwd_model)

    # Calculate data covariance
    if isinstance(data, mne.Epochs):
        data_cov = mne.compute_covariance(data, method="empirical", rank=rank)
    else:
        data_cov = mne.compute_raw_covariance(data, method="empirical", rank=rank)

    # Calculate noise covariance matrix
    #
    # Later this will be inverted and used to whiten the data AND the lead
    # fields as part of the source recon. See:
    #
    #   https://www.sciencedirect.com/science/article/pii/S1053811914010325
    #
    # In MNE, the noise cov is normally obtained from empty room noise
    # recordings or from a baseline period. Here (if no noise cov is passed
    # in) we compute a diagonal noise cov with the variances set to the mean
    # variance of each sensor type (e.g. mag, grad, eeg).
    n_channels = data_cov.data.shape[0]
    noise_cov_diag = np.zeros(n_channels)
    for type in chantypes:
        # Indices of this channel type
        type_data = data.copy().pick(type, exclude="bads")
        inds = []
        for chan in type_data.info["ch_names"]:
            inds.append(data_cov.ch_names.index(chan))

        # Mean variance of channels of this type
        variance = np.mean(np.diag(data_cov.data)[inds])
        noise_cov_diag[inds] = variance
        print(f"Variance for chantype {type} is {variance}")

    bads = [b for b in data.info["bads"] if b in data_cov.ch_names]
    noise_cov = mne.Covariance(
        noise_cov_diag,
        data_cov.ch_names,
        bads,
        data.info["projs"],
        nfree=1e10,
    )

    # Make filters
    filters = mne.beamformer.make_lcmv(
        data.info,
        fwd,
        data_cov,
        noise_cov=noise_cov,
        pick_ori=pick_ori,
        rank=rank,
        reduce_rank=reduce_rank,
        **kwargs,
    )

    print(f"Saving {fns.filters}")
    filters.save(fns.filters, overwrite=True)

    print("LCMV beamformer complete.")


def apply_lcmv_beamformer(
    fns,
    raw,
    reject_by_annotation="omit",
    spatial_resolution=None,
    reference_brain="mni",
):
    """Apply an LCMV beamformer.

    Parameters
    ----------
    fns : OSLFilenames
        Container for OSL filenames.
    raw : instance of mne.io.Raw or mne.Epochs
        The data to apply the LCMV filter to.
    reject_by_annotation : str | list of str | None
        If string, the annotation description to use to reject epochs.
        If list of str, the annotation descriptions to use to reject epochs.
        If None, do not reject epochs.
    spatial_resolution : int, optional
        Resolution to use for the reference brain in mm (must be an integer,
        or will be cast to nearest int). If None, then the gridstep used to
        create the forward model is used.
    reference_brain : str, optional
        Either 'head' or 'mni'.

    Returns
    -------
    voxel_data : np.ndarray
        Voxel data in the space specified by the 'reference_brain' argument.
        Shape is (voxels, time) for a Raw object or (voxels, time, epochs) for
        an Epochs object.
    coords : np.ndarray
        Coordinates for each voxel in the reference brain space.
        Shape is (voxels, 3).
    """
    print()
    print("Applying LCMV beamformer")
    print("------------------------")

    # Load filters
    filters = mne.beamformer.read_beamformer(fns.filters)

    # Pick chantypes that were used to make the beamformer in the data
    raw = raw.copy().pick(filters["ch_names"])

    if isinstance(raw, mne.Epochs):
        # Apply filters to an Epochs object
        stc = mne.beamformer.apply_lcmv_epochs(raw, filters)
        voxel_data_head = np.transpose([s.data for s in stc], axes=[1, 2, 0])
    else:
        # Apply filters to a Raw object
        _check_reference(raw)
        data, times = raw.get_data(
            return_times=True, reject_by_annotation=reject_by_annotation
        )
        chan_inds = mne.utils._check_channels_spatial_filter(raw.ch_names, filters)
        data = data[chan_inds]
        stc = _apply_lcmv(data=data, filters=filters, info=raw.info, tmin=times[0])
        voxel_data_head = next(stc).data

    # Get coordinates in head space
    fwd = mne.read_forward_solution(fns.fwd_model)
    vs = fwd["src"][0]
    voxel_coords_head = vs["rr"][vs["vertno"]] * 1000  # in mm

    if reference_brain == "head":
        return voxel_data_head, voxel_coords_head

    # ------------------------------------------
    # Convert coordinates from head space to MNI
    # ------------------------------------------

    # Convert voxel_coords_head to unscaled MRI
    # head_mri_t_file xform is to unscaled MRI
    head_mri_t = mne.transforms.read_trans(fns.coreg.head_mri_t_file)
    voxel_coords_mri = rhino._xform_points(
        head_mri_t["trans"], voxel_coords_head.T
    ).T

    # Convert voxel_coords_mri to MNI
    # mni_mri_t_file xform is to unscaled MRI
    mni_mri_t = mne.transforms.read_trans(fns.surfaces.mni_mri_t_file)
    voxel_coords_mni = rhino._xform_points(
        np.linalg.inv(mni_mri_t["trans"]), voxel_coords_mri.T
    ).T

    if spatial_resolution is None:
        # Estimate gridstep from forward model
        rr = fwd["src"][0]["rr"]
        spatial_resolution = _get_gridstep(rr)

    spatial_resolution = int(spatial_resolution)
    print(f"spatial_resolution = {spatial_resolution} mm")

    reference_brain = f"{fns.surfaces.fsl_dir}/data/standard/MNI152_T1_1mm_brain.nii.gz"

    # Create standard brain of the required resolution
    reference_brain_resampled = (
        f"{fns.src_dir}/MNI152_T1_{spatial_resolution}mm_brain.nii.gz"
    )
    print(f"mask_file: {reference_brain_resampled}")

    # Get coordinates from reference brain at resolution spatial_resolution
    utils.system_call(
        f"flirt -in {reference_brain} -ref {reference_brain} "
        f"-out {reference_brain_resampled} -applyisoxfm {spatial_resolution}",
        verbose=False
    )
    voxel_coords_mni_resampled, _ = _niimask2mmpointcloud(reference_brain_resampled)

    # For each voxel_coords_mni find nearest coordinate in voxel_coords_head
    print("Finding nearest neighbour in resampled MNI space")
    voxel_data_mni_resampled = np.zeros(
        np.insert(voxel_data_head.shape[1:], 0, voxel_coords_mni_resampled.shape[1])
    )
    for cc in range(voxel_coords_mni_resampled.shape[1]):
        index, dist = _closest_node(voxel_coords_mni_resampled[:, cc], voxel_coords_mni)
        if dist < spatial_resolution:
            voxel_data_mni_resampled[cc] = voxel_data_head[index]

    print("Applying LCMV beamformer complete.")

    return voxel_data_mni_resampled, voxel_coords_mni_resampled


def _niimask2mmpointcloud(nii_mask, volindex=None):
    vol = nib.load(nii_mask).get_fdata()
    if len(vol.shape) == 4 and volindex is not None:
        vol = vol[:, :, :, volindex]
    if not len(vol.shape) == 3:
        Exception(
            "nii_mask must be a 3D volume, or nii_mask must be a 4D volume "
            "with volindex specifying a volume index"
        )
    pc_nativeindex = np.asarray(np.where(vol != 0))
    values = np.asarray(vol[vol != 0])
    pc = rhino._xform_points(rhino._get_sform(nii_mask)["trans"], pc_nativeindex)
    return pc, values


def _closest_node(node, nodes):
    if len(nodes) == 1:
        nodes = np.reshape(nodes, [-1, 1])
    kdtree = KDTree(nodes)
    distance, index = kdtree.query(node)
    return index, distance


def _get_gridstep(coords):
    store = []
    for ii in range(coords.shape[0]):
        store.append(np.sqrt(np.sum(np.square(coords[ii, :] - coords[0, :]))))
    store = np.asarray(store)
    return int(np.round(np.min(store[np.where(store > 0)]) * 1000))


