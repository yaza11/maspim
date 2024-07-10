from typing import Iterable

import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

file = os.path.join(os.path.dirname(__file__),'calibrants.txt')

cal = pd.read_csv(file, sep='\t')

def get_calibrants(
        mz_limits: tuple[float, float],
        calibrants_mz: float | Iterable[float] | None = None
) -> np.ndarray[float]:
    if calibrants_mz is None:
        mz = cal.loc[:, 'm/z']
        mask = (mz > mz_limits[0]) & (mz < mz_limits[1])
        mzs = mz[mask].to_numpy()
        calibrants_mz: list[float] = list(mzs)
        logger.info(f'No calibrants provided, using {calibrants_mz} from data base.')
    if type(calibrants_mz) is float:
        calibrants_mz: list[float] = [calibrants_mz]
    elif not isinstance(calibrants_mz, list):
        calibrants_mz: list[float] = list(calibrants_mz)
    assert all([isinstance(c, float | int) for c in calibrants_mz]), \
        (f'Expected all calibrants to be a number, but found '
         f'{[type(c) for c in calibrants_mz]} for {calibrants_mz}.')
    calibrants_mz.sort()

    return calibrants_mz


if __name__ == '__main__':
    mzs = get_calibrants((388, 490))
