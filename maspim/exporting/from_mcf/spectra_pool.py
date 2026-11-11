"""
This module implements a pool that reads spectra from disk and hands them to worker threads.

One thread reads the spectra in the order in which they are stored on disk (on
external hard drives reading is by far the slowest part) into a queue of
limited size, while worker threads run a task on each spectrum. numpy and scipy
release the GIL in the expensive parts (np.interp, find_peaks, ...), so threads
run tasks in parallel without copying spectra to other processes.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# maximum memory of the spectra waiting in the queue (in bytes)
MAX_BYTES: int = 1024 ** 3  # 1 GB
# mass (in Da) read beyond the limits when spectra are calibrated with functions
# that are not known before reading (linear calibration functions extrapolated
# to the limits can shift a few hundred mDa)
MAX_CALIBRATION_SHIFT: float = 1.

# task(position, mzs, intensities)
Task = Callable[[int, np.ndarray, np.ndarray], None]


def calibration_margin(poly_coeffs: np.ndarray[float] | None, limits: tuple[float, float]) -> float:
    """Largest shift (in Da) of the calibration functions (one per row) at the limits."""
    if (poly_coeffs is None) or (len(poly_coeffs) == 0):
        return 0.
    # np.polyval evaluates all rows at once if the coefficients are columns
    return float(np.max(np.abs([np.polyval(np.asarray(poly_coeffs).T, limit) for limit in limits])))


class ThreadLocalSum:
    """Sum of arrays added from several threads, each thread adds to its own array."""

    def __init__(self, shape: int | tuple[int, ...]) -> None:
        self.shape = shape
        self._local = threading.local()
        self._sums: list[np.ndarray[float]] = []
        self._lock = threading.Lock()

    def add(self, values: np.ndarray[float]) -> None:
        local_sum: np.ndarray[float] | None = getattr(self._local, 'sum', None)
        if local_sum is None:
            local_sum = self._local.sum = np.zeros(self.shape)
            with self._lock:
                self._sums.append(local_sum)
        local_sum += values

    def total(self) -> np.ndarray[float]:
        return np.sum(self._sums, axis=0) if self._sums else np.zeros(self.shape)


class SpectraPool:
    """
    Read spectra in the order they are stored on disk and run a task on each of them.

    Example Usage
    -------------
    >>> pool = SpectraPool(reader, mzs=spec.mzs, limits=spec.limits, n_jobs=8)
    >>> def task(position, mzs, intensities):
    >>>     # runs in a worker thread, position is the position of the spectrum in indices
    >>>     tics[position] = pool.resample(mzs, intensities).sum()
    >>> pool.run(task, spec.indices)
    """

    def __init__(
            self,
            reader: Any,
            mzs: np.ndarray[float],
            limits: tuple[float, float],
            n_jobs: int | None = None,
            max_bytes: int = MAX_BYTES,
            show_progress: bool = False,
            desc: str | None = None,
            callback: Callable[[int, int], None] | None = None
    ) -> None:
        """
        Parameters
        ----------
        reader: RtmsBrukerMCFReader | Hdf5Handler | SpectraCache
            Reader from which to obtain the spectra (with get_spectrum and indices).
        mzs: np.ndarray[float]
            The mz values onto which spectra are resampled (sorted).
        limits: tuple[float, float]
            The mass window.
        n_jobs: int, optional
            Number of worker threads. The default is None, which reads and processes
            spectra one after another in the calling thread. Values smaller than 1
            use all cpus.
        max_bytes: int, optional
            Maximum memory of the spectra waiting in the queue. The default is 1 GB.
        show_progress: bool, optional
            Whether to show a progress bar.
        desc: str, optional
            Description of the progress bar.
        callback: Callable[[int, int], None], optional
            Called with the number of processed spectra and the total number
            after each spectrum (from the worker threads). Exceptions raised by
            the callback stop the pool and are raised by run, e.g. to cancel.
        """
        self.reader = reader
        self.mzs: np.ndarray[float] = np.asarray(mzs)
        self.limits: tuple[float, float] = limits
        self.n_workers: int | None = (
            None if n_jobs is None else (n_jobs if n_jobs > 0 else os.cpu_count())
        )
        self.max_bytes: int = max_bytes
        self.show_progress: bool = show_progress
        self.desc: str | None = desc
        self.callback: Callable[[int, int], None] | None = callback

        # mzs of the last resampled spectrum and the slice of it equal to self.mzs
        self._grid_cache: tuple[np.ndarray[float] | None, slice | None] = (None, None)

    def window(self, margin: float = 0) -> tuple[float, float]:
        """Mass window to read such that spectra shifted by up to margin still cover the limits."""
        # two grid spacings at the limits (the spacing is not constant) ensure
        # there is a data point beyond each limit to interpolate from
        margin += 2 * max(self.mzs[1] - self.mzs[0], self.mzs[-1] - self.mzs[-2])
        return self.limits[0] - margin, self.limits[1] + margin

    def _file_order(self, indices: np.ndarray[int]) -> np.ndarray[int]:
        """Positions in indices sorted by where the spectra are stored on disk."""
        positions: np.ndarray[int] = pd.Index(self.reader.indices).get_indexer(indices)
        assert np.all(positions >= 0), f'indices {indices[positions < 0]} not found in reader'
        # pyrtms knows where spectra are in the mcf file, other readers store them in order
        raw_positions: np.ndarray[int] | None = getattr(self.reader, '_raw_positions', None)
        keys: np.ndarray[int] = positions if raw_positions is None else np.asarray(raw_positions)[positions]
        return np.argsort(keys, kind='stable')

    def run(self, task: Task, indices: Iterable[int], margin: float = 0) -> None:
        """
        Call task(position, mzs, intensities) for every spectrum in indices.

        Parameters
        ----------
        task: Callable[[int, np.ndarray, np.ndarray], None]
            Function receiving the position of the spectrum in indices and its mzs
            and intensities within the window (see window). With n_jobs, tasks run
            in several threads at the same time, so they should only write to rows
            of arrays belonging to their position or add to a ThreadLocalSum.
        indices: Iterable[int]
            Indices of the spectra (as used by the reader).
        margin: float, optional
            Mass (in Da) to read beyond the limits, e.g. the largest calibration
            shift (see calibration_margin).
        """
        indices: np.ndarray[int] = np.asarray(indices)
        window: tuple[float, float] = self.window(margin)
        order: np.ndarray[int] = self._file_order(indices)

        def read(k: int) -> tuple[int, np.ndarray[float], np.ndarray[float]]:
            mzs, intensities = self.reader.get_spectrum(int(indices[k]), return_mzs=True, limits=window)
            return k, mzs, intensities

        progress = tqdm(
            total=len(indices), desc=self.desc, disable=not self.show_progress,
            smoothing=50 / max(len(indices), 1)
        )
        self._n_done, self._n_total = 0, len(indices)
        try:
            if self.n_workers is None:
                for k in order:
                    task(*read(k))
                    self._update(progress)
            elif len(order) > 0:
                self._run_threads(task, read, order, progress)
        finally:
            progress.close()

    def _run_threads(
            self,
            task: Task,
            read: Callable[[int], tuple[int, np.ndarray[float], np.ndarray[float]]],
            order: np.ndarray[int],
            progress: tqdm
    ) -> None:
        """One thread reads spectra into a queue, n_workers threads run the task on them."""
        first = read(order[0])
        # limit the memory of spectra waiting in the queue
        spectra: queue.Queue = queue.Queue(maxsize=max(1, self.max_bytes // max(first[2].nbytes, 1)))
        stop: threading.Event = threading.Event()
        errors: list[BaseException] = []
        lock: threading.Lock = threading.Lock()

        def produce() -> None:
            try:
                spectra.put(first)
                for k in order[1:]:
                    if stop.is_set():
                        break
                    spectra.put(read(k))
            except BaseException as e:
                errors.append(e)
                stop.set()
            finally:
                # one end marker per worker
                for _ in range(self.n_workers):
                    spectra.put(None)

        def consume() -> None:
            while (item := spectra.get()) is not None:
                # after an error, only empty the queue such that the reader can finish
                if stop.is_set():
                    continue
                try:
                    task(*item)
                    with lock:
                        self._update(progress)
                except BaseException as e:
                    errors.append(e)
                    stop.set()

        threads: list[threading.Thread] = [threading.Thread(target=produce, daemon=True)] + [
            threading.Thread(target=consume, daemon=True) for _ in range(self.n_workers)
        ]
        for thread in threads:
            thread.start()
        try:
            for thread in threads:
                # join with timeout, so that KeyboardInterrupt is not blocked
                while thread.is_alive():
                    thread.join(.5)
        except KeyboardInterrupt:
            stop.set()
            raise
        if errors:
            raise errors[0]

    def _update(self, progress: tqdm) -> None:
        progress.update()
        # count separately, a disabled progress bar does not count
        self._n_done += 1
        if self.callback is not None:
            self.callback(self._n_done, self._n_total)

    def _on_grid(self, mzs_spectrum: np.ndarray[float]) -> slice | None:
        """Slice of mzs_spectrum equal to self.mzs or None, if there is none."""
        cached_mzs, cached_slice = self._grid_cache
        # readers return the same (cached) array for spectra with the same calibration
        if mzs_spectrum is cached_mzs:
            return cached_slice
        start: int = int(np.searchsorted(mzs_spectrum, self.mzs[0]))
        candidate: np.ndarray[float] = mzs_spectrum[start:start + len(self.mzs)]
        on_grid: slice | None = (
            slice(start, start + len(self.mzs))
            if (len(candidate) == len(self.mzs)) and np.allclose(candidate, self.mzs, rtol=1e-12, atol=0)
            else None
        )
        self._grid_cache = (mzs_spectrum, on_grid)
        return on_grid

    def resample(
            self,
            mzs: np.ndarray[float],
            intensities: np.ndarray[float],
            poly_coeffs: np.ndarray[float] | None = None
    ) -> np.ndarray[float]:
        """
        Calibrate a spectrum (if poly_coeffs are given and not all 0) and return
        its intensities at self.mzs.

        Uncalibrated spectra whose mzs contain self.mzs are not interpolated, the
        returned array is then a view of intensities.
        """
        if (poly_coeffs is None) or (not np.any(poly_coeffs)):
            on_grid: slice | None = self._on_grid(mzs)
            if on_grid is not None:
                return intensities[on_grid]
        else:
            # same convention as apply_calibration_mz
            mzs = mzs + np.polyval(poly_coeffs, mzs)
        return np.interp(self.mzs, mzs, intensities)
