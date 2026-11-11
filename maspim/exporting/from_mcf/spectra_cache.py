"""
This module implements a cache of spectra on a fast local disk.

Measurements usually live on external hard drives which read ~40 MB/s, so reading
all spectra of a measurement takes about half an hour, and the steps of the
Spectra class read all spectra several times. The cache stores the spectra within
the mass window (plus a margin for calibration) in a file on a local SSD while
they are read for the first time; afterward, the cache is used as the reader.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd

from maspim.exporting.from_mcf.spectra_pool import MAX_CALIBRATION_SHIFT

logger = logging.getLogger(__name__)


def _source_name(reader: Any) -> str:
    """Name of the d-folder of a reader (copies of a measurement on other drives match)."""
    path: str = getattr(reader, 'mcfdir', None) or getattr(reader, 'path_folder', None) or ''
    return os.path.basename(os.path.normpath(path))


def _crop(mzs: np.ndarray[float], limits: tuple[float, float]) -> slice:
    """Slice of mzs with limits[0] <= mz <= limits[1] (same bounds as RtmsBrukerMCFReader)."""
    return slice(
        int(np.searchsorted(mzs, limits[0], side='left')),
        int(np.searchsorted(mzs, limits[1], side='right'))
    )


class SpectraCache:
    """
    Spectra within a mass window stored in a local file, with the reader interface
    (get_spectrum, indices, mzs) used by the Spectra class.

    Spectra are stored as float32 on the mz values of the first spectrum, so for
    a measurement where all spectra have the same mz values (single calibration)
    the cache holds exact copies.

    Usually, the cache is managed by the Spectra class in a temporary folder,
    which is deleted with the Spectra object
    >>> spec.set_temp_dir(r'C:\\local\\ssd')
    >>> spec.full(reader, n_jobs=8)

    Example Usage (persistent cache)
    --------------------------------
    >>> cache = SpectraCache(r'C:\\local\\ssd\\greifensee', limits=spec.limits)
    Returns the cache if it is complete, otherwise a reader that fills the cache
    while spectra are read for the first step and switches to the cache once it is complete
    >>> reader = cache.cached(reader)
    >>> spec.full(reader, n_jobs=8)
    """
    all_mz_values_same: bool = True

    def __init__(
            self,
            path_folder: str,
            limits: tuple[float, float] | None = None,
            margin: float = MAX_CALIBRATION_SHIFT
    ) -> None:
        """
        Parameters
        ----------
        path_folder: str
            Folder of the cache files (should be on a fast local disk).
        limits: tuple[float, float], optional
            Mass window to store. Only needed to create the cache.
        margin: float, optional
            Mass (in Da) stored beyond the limits, spectra calibrated by up to
            this shift can be resampled without extrapolating. The default is 1 Da.
        """
        self.path_folder: str = path_folder
        self.path_intensities: str = os.path.join(path_folder, 'intensities.npy')
        # written once all spectra are in the cache
        self.path_meta: str = os.path.join(path_folder, 'meta.npz')
        self.limits: tuple[float, float] | None = limits
        self.margin: float = margin

        self.indices: np.ndarray[int] | None = None
        self.mzs: np.ndarray[float] | None = None
        # stored mass window (mzs may not reach it, if the spectra end before)
        self.window: tuple[float, float] | None = None
        self.source: str | None = None
        self._intensities: np.ndarray[np.float32] | None = None
        self._rows: pd.Index | None = None
        self._written: np.ndarray[bool] | None = None
        # slices of self.mzs for requested limits, mzs are returned as the same array for the same limits
        self._windows: dict[tuple[float, float], tuple[slice, np.ndarray[float]]] = {}
        self._warned_extrapolation: bool = False

        if os.path.exists(self.path_meta):
            self._load()

    @property
    def is_complete(self) -> bool:
        return os.path.exists(self.path_meta) and (self._written is None)

    def _load(self) -> None:
        with np.load(self.path_meta) as meta:
            self.indices = meta['indices']
            self.mzs = meta['mzs']
            if self.limits is None:
                self.limits = tuple(meta['limits'])
            self.window = tuple(meta['window'])
            self.source = str(meta['source'])
        self._intensities = np.load(self.path_intensities, mmap_mode='r')
        self._rows = pd.Index(self.indices)
        self._windows = {}

    def matches(self, reader: Any, indices: np.ndarray[int] | None = None) -> bool:
        """Whether the cache is complete and holds the spectra of reader with indices (default: all)."""
        if not self.is_complete:
            return False
        if indices is None:
            indices = reader.indices
        return (_source_name(reader) == self.source) and np.array_equal(np.asarray(indices), self.indices)

    def covers(self, limits: tuple[float, float]) -> bool:
        """Whether the stored mass window contains limits."""
        return (self.window is not None) and (self.window[0] <= limits[0]) and (limits[1] <= self.window[1])

    def cached(self, reader: Any, indices: np.ndarray[int] | None = None) -> SpectraCache | CachingReader:
        """
        Return the cache if it holds the spectra of reader with indices (default: all)
        for self.limits, otherwise a reader that writes the spectra to the cache
        while reading them.
        """
        if self.matches(reader, indices) and self.covers(self.limits):
            logger.info(f'using cached spectra in {self.path_folder}')
            return self
        assert self.limits is not None, 'provide limits to create the cache'
        return CachingReader(reader, self, indices)

    def close(self) -> None:
        """Release the cache file (arrays returned by get_spectrum keep it open)."""
        self._intensities = None
        self._written = None

    def remove(self) -> None:
        """Delete the cache folder."""
        self.close()
        shutil.rmtree(self.path_folder, ignore_errors=True)
        if os.path.exists(self.path_folder):
            logger.warning(f'could not delete {self.path_folder}, is the cache still in use?')

    def _create(
            self, reader: Any, indices: np.ndarray[int], mzs: np.ndarray[float], window: tuple[float, float]
    ) -> None:
        """Create the file for the spectra of reader with indices within window, sampled at mzs."""
        os.makedirs(self.path_folder, exist_ok=True)
        size: int = len(indices) * len(mzs) * np.dtype(np.float32).itemsize
        free: int = shutil.disk_usage(self.path_folder).free
        if size > free:
            raise OSError(
                f'cache needs {size / 1024 ** 3:.1f} GB, but only {free / 1024 ** 3:.1f} GB '
                f'are free in {self.path_folder}'
            )
        logger.info(f'creating cache with {size / 1024 ** 3:.1f} GB in {self.path_folder}')
        if os.path.exists(self.path_meta):
            os.remove(self.path_meta)
        # release an opened (old) cache file before overwriting it
        self._intensities = None
        self._intensities = np.lib.format.open_memmap(
            self.path_intensities, mode='w+', dtype=np.float32, shape=(len(indices), len(mzs))
        )
        self.indices = indices
        self.mzs = np.asarray(mzs, dtype=float)
        self.window = window
        self.source = _source_name(reader)
        self._rows = pd.Index(indices)
        self._written = np.zeros(len(indices), dtype=bool)
        self._windows = {}

    def _write(self, index: int, mzs: np.ndarray[float], intensities: np.ndarray[float]) -> None:
        """Store a spectrum read from the source (mzs must cover self.mzs)."""
        if index not in self._rows:
            return
        row: int = self._rows.get_loc(index)
        start: int = int(np.searchsorted(mzs, self.mzs[0]))
        candidate: np.ndarray[float] = mzs[start:start + len(self.mzs)]
        if (len(candidate) == len(self.mzs)) and np.allclose(candidate, self.mzs, rtol=1e-12, atol=0):
            self._intensities[row] = intensities[start:start + len(self.mzs)]
        else:  # spectra with a different calibration (mz values)
            self._intensities[row] = np.interp(self.mzs, mzs, intensities)
        self._written[row] = True
        if self._written.all():
            self._finish()

    def _finish(self) -> None:
        """Flush the spectra and write the metadata, which marks the cache as complete."""
        self._intensities.flush()
        np.savez(
            self.path_meta, indices=self.indices, mzs=self.mzs,
            limits=np.asarray(self.limits), window=np.asarray(self.window), source=self.source
        )
        self._written = None
        # reopen read-only
        self._intensities = None
        self._load()
        logger.info(f'cache in {self.path_folder} is complete')

    def get_spectrum(
            self,
            index: int,
            return_mzs: bool = None,
            intensities_only: bool = None,
            limits: tuple[float, float] = None
    ) -> tuple[np.ndarray[float] | None, np.ndarray[np.float32]]:
        """
        Return the spectrum with the given index (same signature as RtmsBrukerMCFReader.get_spectrum).

        The intensities are a read-only view of the cache file.
        """
        assert self.is_complete, 'cache is not complete'
        if (return_mzs is not None) and (intensities_only is not None):
            raise ValueError('Provide only one of return_mzs and intensity_only, not both.')
        if return_mzs is None:
            return_mzs = (not intensities_only) if (intensities_only is not None) else True

        if limits is None:
            window, mzs = slice(None), self.mzs
        else:
            limits = (float(limits[0]), float(limits[1]))
            if limits not in self._windows:
                if (not self.covers(limits)) and (not self._warned_extrapolation):
                    logger.warning(
                        f'requested mass window {limits} exceeds the cached window '
                        f'{self.window}, spectra near the limits will be extrapolated. '
                        f'Create the cache with a larger margin.'
                    )
                    self._warned_extrapolation = True
                window = _crop(self.mzs, limits)
                self._windows[limits] = (window, self.mzs[window])
            window, mzs = self._windows[limits]

        intensities: np.ndarray[np.float32] = self._intensities[self._rows.get_loc(int(index)), window]
        return (mzs if return_mzs else None), intensities


class CachingReader:
    """
    Reader that writes the spectra it reads to a SpectraCache and reads from the
    cache once all spectra are in it (see SpectraCache.cached).

    Other attributes are those of the wrapped reader.
    """

    def __init__(self, reader: Any, cache: SpectraCache, indices: np.ndarray[int] | None = None) -> None:
        """
        Parameters
        ----------
        reader: RtmsBrukerMCFReader | Hdf5Handler
            Reader from which to obtain the spectra.
        cache: SpectraCache
            Cache to fill.
        indices: np.ndarray[int], optional
            Indices of the spectra to store. The default is all spectra of reader,
            other spectra are always read from reader.
        """
        self.reader = reader
        self.cache: SpectraCache = cache
        self._indices: np.ndarray[int] = np.asarray(reader.indices if indices is None else indices)
        # stored mass window, extended by the first requested window (see get_spectrum)
        self._window: tuple[float, float] = (cache.limits[0] - cache.margin, cache.limits[1] + cache.margin)
        self._created: bool = False
        # slices of mzs arrays for requested limits (key: id of the mzs array, limits)
        self._crops: dict[tuple[int, tuple[float, float]], tuple[np.ndarray[float], slice, np.ndarray[float]]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.reader, name)

    def _crop(self, mzs: np.ndarray[float], limits: tuple[float, float]) -> tuple[slice, np.ndarray[float]]:
        """Crop mzs to limits, returning the same array for the same mzs and limits."""
        key = (id(mzs), limits)
        cached = self._crops.get(key)
        if (cached is None) or (cached[0] is not mzs):
            if len(self._crops) > 16:  # mzs differ between spectra
                self._crops.clear()
            window: slice = _crop(mzs, limits)
            cached = self._crops[key] = (mzs, window, mzs[window])
        return cached[1], cached[2]

    def get_spectrum(
            self,
            index: int,
            return_mzs: bool = None,
            intensities_only: bool = None,
            limits: tuple[float, float] = None
    ) -> tuple[np.ndarray[float] | None, np.ndarray[float]]:
        """Read a spectrum from the reader and store it in the cache (same signature as get_spectrum)."""
        if self.cache.is_complete and (int(index) in self.cache._rows):
            return self.cache.get_spectrum(
                index, return_mzs=return_mzs, intensities_only=intensities_only, limits=limits
            )
        if (return_mzs is not None) and (intensities_only is not None):
            raise ValueError('Provide only one of return_mzs and intensity_only, not both.')
        if return_mzs is None:
            return_mzs = (not intensities_only) if (intensities_only is not None) else True

        if not self._created:
            # store at least the first requested window (it contains the edges for interpolation)
            if limits is not None:
                self._window = (min(self._window[0], float(limits[0])), max(self._window[1], float(limits[1])))
            mzs, _ = self.reader.get_spectrum(int(self._indices[0]), return_mzs=True, limits=self._window)
            # like SpectraPool.window, two grid spacings beyond the margin to interpolate from
            edge: float = 2 * max(mzs[1] - mzs[0], mzs[-1] - mzs[-2])
            self._window = (self._window[0] - edge, self._window[1] + edge)
            mzs, _ = self.reader.get_spectrum(int(self._indices[0]), return_mzs=True, limits=self._window)
            self.cache._create(self.reader, self._indices, mzs, self._window)
            self._created = True

        # read at least the stored window
        read_limits: tuple[float, float] | None = None if limits is None else (
            min(self._window[0], float(limits[0])), max(self._window[1], float(limits[1]))
        )
        mzs, intensities = self.reader.get_spectrum(int(index), return_mzs=True, limits=read_limits)
        if not self.cache.is_complete:
            self.cache._write(int(index), mzs, intensities)
        if limits is not None:
            window, mzs = self._crop(mzs, (float(limits[0]), float(limits[1])))
            intensities = intensities[window]
        return (mzs if return_mzs else None), intensities
