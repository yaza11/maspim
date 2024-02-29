import numpy as np
import pandas as pd


def extract_mz(target_mz, spec, tol=0.01, min_int=10000, min_snr=0):
    """
    extract one m/z value from one spectrum
    :param target_mz:   the m/z value to be extracted
    :param spec:       the spectrum
    :param tol:       the tolerance for m/z values
    :param min_int:  the minimum intensity
    :param min_snr: the minimum signal-to-noise ratio
    :return:       the m/z value, intensity and signal-to-noise ratio
    """
    # sort the spectrum by intensity
    spec = spec[spec[:, 1].argsort()[::-1]]
    # find the index of the highest intensity peak within the tolerance
    spec = spec[(spec[:, 0] >= target_mz - tol / 2)
                & (spec[:, 0] <= target_mz + tol / 2)
                & (spec[:, 1] >= min_int)
                & (spec[:, 2] >= min_snr)]
    if spec.shape[0] == 0:
        return np.nan, np.nan, np.nan
    else:
        return spec[0, 0], spec[0, 1], spec[0, 2]


def extract_mzs(target_mz, txt_path, tol=0.01, min_int=10000, min_snr=0):
    """
    this is the legacy function to read the txt file exported from Bruker DataAnalysis, and
    extract the target m/z values and intensities for all spectra from the txt file
    :param target_mz: the m/z values to be extracted, either a list of m/z values or a single m/z value
    :param txt_path: the path to the txt file
    :param tol: the tolerance for m/z values
    :param min_int: the minimum intensity
    :param min_snr: the minimum signal-to-noise ratio
    :return: a dataframe containing the target m/z values and intensities for all spectra, and the ppm error for each
    """
    # preprocessing target_mz if it's a dictionary
    if isinstance(target_mz, dict):
        mz_names = list(target_mz.keys())
        target_mz = list(target_mz.values())
    elif isinstance(target_mz, float):
        target_mz = [target_mz]
        mz_names = [0]
    else:
        mz_names = range(len(target_mz))

    # convert them to strings
    mz_names = [str(mz) for mz in mz_names]

    # read the txt file
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    # get all spectra, each spectrum is a line and starts with 'R**X**Y**'
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line.startswith('R')]

    # get the spot name for each spectrum
    spot_names = [line.split(';')[0] for line in lines]
    # get the m/z values for each spectrum, it's after every two ';'
    mzs = [line.split(';')[2::3] for line in lines]
    # convert everything to floats
    mzs = [[round(float(mz), 4) for mz in mz_list] for mz_list in mzs]
    intensities = [line.split(';')[3::3] for line in lines]
    intensities = [[float(intensity) for intensity in intensity_list] for intensity_list in intensities]
    snrs = [line.split(';')[4::3] for line in lines]
    snrs = [[float(snr) for snr in snr_list] for snr_list in snrs]
    mz = np.zeros((len(spot_names), len(target_mz)))
    intensity = np.zeros((len(spot_names), len(target_mz)))
    snr = np.zeros((len(spot_names), len(target_mz)))
    for i in range(len(spot_names)):
        spec = np.array([mzs[i], intensities[i], snrs[i]]).T
        for j in range(len(target_mz)):
            mz[i, j], intensity[i, j], snr[i, j] = extract_mz(target_mz[j], spec, tol, min_int, min_snr)
    df = pd.DataFrame({'spot_name': spot_names})
    for i in range(len(target_mz)):
        df['mz_' + mz_names[i]] = mz[:, i]
        df['Int_' + mz_names[i]] = intensity[:, i]
        df['snr_' + mz_names[i]] = snr[:, i]
    return df


def extract_all(txt_path, min_int=10000, min_snr=1, peak_th=0.1, min_member=0.1, tol=10):
    # test if module 'mfe' is installed
    try:
        import mfe
        from mfe.from_txt import msi_from_txt, get_ref_peaks, create_feature_table
    except ImportError:
        raise ImportError(
            'Please install mfe first by running "pip install git+https://github.com/weimin-liu/msi_feature_extraction.git"')

    spectra = msi_from_txt(raw_txt_path=txt_path, min_int=min_int, min_snr=min_snr)

    ref = get_ref_peaks(spectra, peak_th=peak_th, min_member=min_member, tol=tol)

    feature_table = create_feature_table(spectra, ref[peak_th], tol=tol)

    return feature_table


if __name__ == "__main__":
    pass
