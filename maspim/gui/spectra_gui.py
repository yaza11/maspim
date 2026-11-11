"""
This module implements a GUI for selecting the parameters of the Spectra class interactively.

Steps working on the summed spectrum (noise level, peaks, filtering of binned
intensities) are updated as soon as a parameter changes. Steps reading all
spectra (summing, calibration, binning) run in the background with a progress
bar and can be cancelled. For the calibration, a few random spectra can be
previewed before calibrating all spectra.

Requires PyQt5 (pip install maspim[gui]).

Example Usage
-------------
Open the window and select the measurement in it, from the command line
$ python -m maspim.gui.spectra_gui
(or maspim-spectra-gui, if maspim is installed), or from Python
>>> from maspim.gui.spectra_gui import run_spectra_gui
>>> run_spectra_gui()

Open the window for a measurement that is already loaded, it blocks until the
window is closed (unless a Qt event loop is running, e.g. in IPython with
%gui qt), spectra keeps the results of the selected parameters
>>> reader = RtmsBrukerMCFReader.from_dir('path/to/folder.i/folder.d')
>>> spectra = Spectra.from_reader(reader)
>>> run_spectra_gui(spectra, reader, path_temp='path/to/fast/disk')
"""
from __future__ import annotations

import copy
import logging
import os
import sys
import traceback
from typing import Any, Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy.signal import find_peaks, peak_widths
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError as e:
    raise ImportError('the GUI requires PyQt5, install it with pip install maspim[gui]') from e

from pyrtms.rtmsBrukerMCFReader import RtmsBrukerMCFReader

from maspim.exporting.from_mcf.helper import find_polycalibration_spectrum
from maspim.exporting.from_mcf.spectrum import Spectra
from maspim.res.calibrants import get_calibrants
from maspim.util.convenience import check_attr

logger = logging.getLogger(__name__)

# the visible part of a spectrum is drawn as the minimum and maximum of this many bins
N_PLOT_BINS: int = 2000
# dpi of the figures in the window (Qt scales them for high-resolution displays), the
# default of matplotlib may be set for printing (e.g. 300 by imported packages)
SCREEN_DPI: float = 100
# at most this many peaks are drawn (the highest visible ones)
N_PLOT_PEAKS: int = 3000
# spectrum attributes with one entry per peak after finding peaks (restored before fitting kernels)
PEAK_STEP_ATTRS: tuple[str, ...] = (
    '_peaks', '_peak_properties', '_peak_setting_parameters', '_peaks_SNR', '_peaks_is_side_peak'
)
# spectrum attributes with one entry per peak after binning (restored before filtering binned intensities)
PEAK_ATTRS: tuple[str, ...] = PEAK_STEP_ATTRS + ('_kernel_params', '_line_spectra')
# steps whose results depend on all previous ones (calibrating replaces the summed spectrum)
STAGES: tuple[str, ...] = ('sum', 'noise', 'peaks', 'kernels', 'binning')
STAGE_LABELS: dict[str, str] = {
    'sum': 'summed spectrum', 'noise': 'noise level', 'calibration': 'calibration', 'peaks': 'peaks',
    'kernels': 'kernels', 'binning': 'binned intensities', 'filter': 'filter of the binned intensities'
}
# results that take long to compute (the GUI asks before discarding them)
EXPENSIVE: tuple[str, ...] = ('kernels', 'binning')
# colors of the status of steps
COLORS: dict[str, str] = {'done': '#2e7d32', 'not run': '#757575', 'changed': '#e65100'}


def _join(names) -> str:
    """'a', 'a and b' or 'a, b and c'."""
    names = list(names)
    return ' and '.join([', '.join(names[:-1]), names[-1]]) if len(names) > 1 else ''.join(names)


def fwhm_near(
        mzs: np.ndarray[float], intensities: np.ndarray[float], center: float, half_window: float
) -> tuple[float, float] | None:
    """Position and full width at half maximum (in Da) of the highest peak within center +/- half_window."""
    start, stop = np.searchsorted(mzs, [center - half_window, center + half_window])
    if stop - start < 3:
        return None
    x: np.ndarray[float] = mzs[start:stop]
    y: np.ndarray[float] = intensities[start:stop] - intensities[start:stop].min()
    peaks: np.ndarray[int] = find_peaks(y)[0]
    if len(peaks) == 0:
        return None
    peak: int = int(peaks[np.argmax(y[peaks])])
    _, _, left, right = peak_widths(y, [peak], rel_height=.5)
    # fractional indices to mz values
    left_mz, right_mz = np.interp([left[0], right[0]], np.arange(len(x)), x)
    return float(x[peak]), float(right_mz - left_mz)


def envelope(
        x: np.ndarray[float], y: np.ndarray[float], limits: tuple[float, float], n_bins: int = N_PLOT_BINS
) -> tuple[np.ndarray[float], np.ndarray[float]]:
    """Points of (x, y) within limits, reduced to the minimum and maximum of n_bins bins."""
    start: int = max(int(np.searchsorted(x, limits[0])) - 1, 0)
    stop: int = min(int(np.searchsorted(x, limits[1])) + 1, len(x))
    x, y = x[start:stop], y[start:stop]
    if len(x) <= 2 * n_bins:
        return x, y
    size: int = len(x) // n_bins
    n: int = size * n_bins
    xb, yb = x[:n].reshape(n_bins, size), y[:n].reshape(n_bins, size)
    i_min, i_max = yb.argmin(axis=1), yb.argmax(axis=1)
    # keep minimum and maximum in the order of their mz values
    first, second = np.minimum(i_min, i_max), np.maximum(i_min, i_max)
    rows: np.ndarray[int] = np.arange(n_bins)
    return (np.column_stack([xb[rows, first], xb[rows, second]]).ravel(),
            np.column_stack([yb[rows, first], yb[rows, second]]).ravel())


def find_d_folder(path_folder: str) -> str:
    """Return path_folder if it is a d-folder, otherwise the only d-folder inside it."""
    path_folder = os.path.normpath(path_folder)
    if not os.path.isdir(path_folder):
        raise NotADirectoryError(f'{path_folder!r} is not a folder')
    if path_folder.endswith('.d'):
        return path_folder
    d_folders: list[str] = [
        name for name in os.listdir(path_folder)
        if name.endswith('.d') and os.path.isdir(os.path.join(path_folder, name))
    ]
    if len(d_folders) != 1:
        raise FileNotFoundError(
            f'{path_folder!r} contains {len(d_folders)} d-folders ({", ".join(d_folders)}), '
            f'select the d-folder of the measurement' if d_folders
            else f'{path_folder!r} does not contain a d-folder'
        )
    return os.path.join(path_folder, d_folders[0])


def _spin(value: int, minimum: int, maximum: int, tooltip: str = '') -> QtWidgets.QSpinBox:
    spin = QtWidgets.QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(value)
    spin.setToolTip(tooltip)
    return spin


def _dspin(
        value: float, minimum: float, maximum: float, step: float, decimals: int, tooltip: str = ''
) -> QtWidgets.QDoubleSpinBox:
    spin = QtWidgets.QDoubleSpinBox()
    # always use a decimal point
    spin.setLocale(QtCore.QLocale.c())
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    spin.setToolTip(tooltip)
    return spin


def _check(checked: bool, text: str, tooltip: str = '') -> QtWidgets.QCheckBox:
    check = QtWidgets.QCheckBox(text)
    check.setChecked(checked)
    check.setToolTip(tooltip)
    return check


class FloatEdit(QtWidgets.QLineEdit):
    """Line edit for floats in any notation (e.g. 1e5), emits valueChanged when editing is finished."""
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, value: float, tooltip: str = '') -> None:
        super().__init__(f'{value:g}')
        self._value: float = value
        self.setToolTip(tooltip)
        self.editingFinished.connect(self._on_edit)

    def value(self) -> float:
        return self._value

    def _on_edit(self) -> None:
        try:
            value: float = float(self.text())
        except ValueError:
            self.setText(f'{self._value:g}')
            return
        if value != self._value:
            self._value = value
            self.valueChanged.emit(value)


class Cancelled(Exception):
    pass


class Worker(QtCore.QThread):
    """Run a function in a background thread."""
    progress = QtCore.pyqtSignal(int, int)
    failed = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(object)

    def __init__(self, func: Callable[[], Any], parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self.func: Callable[[], Any] = func
        self.cancelled: bool = False

    def run(self) -> None:
        try:
            result: Any = self.func()
        except Cancelled:
            self.failed.emit('cancelled')
        except Exception:
            self.failed.emit(traceback.format_exc())
        else:
            self.done.emit(result)

    def report(self, n: int, total: int) -> None:
        """Progress callback for Spectra (called from worker threads), cancels by raising."""
        if self.cancelled:
            raise Cancelled()
        self.progress.emit(n, total)


class _LogSignal(QtCore.QObject):
    message = QtCore.pyqtSignal(str)


class LogHandler(logging.Handler):
    """Forward warnings of maspim to the status bar (also from worker threads)."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.signal: _LogSignal = _LogSignal()

    def emit(self, record: logging.LogRecord) -> None:
        self.signal.message.emit(record.getMessage())


class SpectrumPlot(QtWidgets.QWidget):
    """
    Summed spectrum with baseline, peaks, reconstruction and binned intensities.

    Only the visible part of the curves is drawn (reduced to N_PLOT_BINS bins),
    so zooming stays fast for spectra with millions of points. Clicking a peak
    (outside the zoom and pan modes of the toolbar) emits peakClicked.
    """
    # mz of the clicked peak
    peakClicked = QtCore.pyqtSignal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure: Figure = Figure(layout='constrained', dpi=SCREEN_DPI)
        self.canvas: FigureCanvasQTAgg = FigureCanvasQTAgg(self.figure)
        self.toolbar: NavigationToolbar2QT = NavigationToolbar2QT(self.canvas, self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot()
        self.ax.set_xlabel('m/z in Da')
        self.ax.set_ylabel('summed intensity')
        curve_styles: dict[str, dict[str, Any]] = {
            'uncalibrated': dict(color='0.6', linewidth=.8, label='uncalibrated sum'),
            'spectrum': dict(color='C0', linewidth=.8, label='summed spectrum'),
            'baseline': dict(color='C1', linewidth=1, linestyle='--', label='baseline'),
            'reconstructed': dict(color='C2', linewidth=.8, label='kernels'),
        }
        self.curves: dict[str, tuple[np.ndarray[float], np.ndarray[float]] | None] = dict.fromkeys(curve_styles)
        self.lines = {name: self.ax.plot([], [], **style)[0] for name, style in curve_styles.items()}
        marker_styles: dict[str, dict[str, Any]] = {
            'removed': dict(marker='x', color='0.6', s=20, label='peaks removed by kernel fit'),
            'peaks': dict(marker='x', color='C3', s=20, label='peaks'),
            'prominent': dict(marker='o', facecolors='none', edgecolors='C1', s=70, linewidths=1.2,
                              label='peaks of sigma estimate'),
            'selected': dict(marker='o', facecolors='none', edgecolors='k', s=140, linewidths=1.5,
                             label='selected peak'),
        }
        self.markers = {name: self.ax.scatter([], [], zorder=3, **style) for name, style in marker_styles.items()}
        self.marker_data: dict[str, tuple[np.ndarray[float], np.ndarray[float]] | None] = dict.fromkeys(marker_styles)
        self.binned = LineCollection([], colors='C4', linewidths=1.5, label='binned (summed)')
        self.ax.add_collection(self.binned)
        # mz, bottom, top
        self.binned_data: tuple[np.ndarray[float], np.ndarray[float], np.ndarray[float]] | None = None

        self._timer = QtCore.QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._draw_visible)
        self.ax.callbacks.connect('xlim_changed', lambda _: self._timer.start())
        self.canvas.mpl_connect('button_press_event', self._on_click)

    def set_curve(self, name: str, x: np.ndarray[float] | None, y: np.ndarray[float] | None = None) -> None:
        self.curves[name] = None if x is None else (np.asarray(x), np.asarray(y))

    def set_markers(self, name: str, x: np.ndarray[float] | None, y: np.ndarray[float] | None = None) -> None:
        self.marker_data[name] = None if (x is None) or (len(x) == 0) else (np.asarray(x), np.asarray(y))

    def set_binned(self, x: np.ndarray[float] | None, bottom=None, top=None) -> None:
        self.binned_data = None if x is None else (np.asarray(x), np.asarray(bottom), np.asarray(top))

    def clear(self) -> None:
        for name in self.curves:
            self.set_curve(name, None)
        for name in self.marker_data:
            self.set_markers(name, None)
        self.set_binned(None)

    def reset_view(self) -> None:
        """Show the entire spectrum."""
        if self.curves['spectrum'] is None:
            return
        x, y = self.curves['spectrum']
        self.ax.set_xlim(x[0], x[-1])
        self.ax.set_ylim(0, 1.05 * max(float(y.max()), 1))
        # the home button of the toolbar returns to this view
        self.toolbar.update()
        self.update_plot()

    def update_plot(self) -> None:
        self._draw_visible()
        handles = [line for name, line in self.lines.items() if self.curves[name] is not None]
        handles += [markers for name, markers in self.markers.items() if self.marker_data[name] is not None]
        if self.binned_data is not None:
            handles.append(self.binned)
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        if handles:
            self.ax.legend(handles=handles, loc='upper right')

    def _draw_visible(self) -> None:
        limits: tuple[float, float] = self.ax.get_xlim()
        for name, line in self.lines.items():
            line.set_data(*(([], []) if self.curves[name] is None else envelope(*self.curves[name], limits)))

        for name, markers in self.markers.items():
            offsets: np.ndarray[float] = np.empty((0, 2))
            if self.marker_data[name] is not None:
                x, y = self.marker_data[name]
                visible: np.ndarray[int] = np.flatnonzero((x >= limits[0]) & (x <= limits[1]))
                visible = visible[np.argsort(y[visible])[::-1][:N_PLOT_PEAKS]]
                offsets = np.column_stack([x[visible], y[visible]])
            markers.set_offsets(offsets)

        segments: list[np.ndarray[float]] = []
        if self.binned_data is not None:
            x, bottom, top = self.binned_data
            visible = (x >= limits[0]) & (x <= limits[1])
            segments = [np.array([[xi, b], [xi, t]]) for xi, b, t in zip(x[visible], bottom[visible], top[visible])]
        self.binned.set_segments(segments)
        self.canvas.draw_idle()

    def _on_click(self, event) -> None:
        mode = getattr(self.toolbar.mode, 'name', self.toolbar.mode)
        if (event.inaxes is not self.ax) or (event.button != 1) or (mode not in ('NONE', '')):
            return
        best: tuple[float, float] | None = None  # (distance in pixels, mz)
        for name in ('peaks', 'removed'):
            offsets: np.ndarray[float] = np.asarray(self.markers[name].get_offsets())
            if len(offsets) == 0:
                continue
            pixels: np.ndarray[float] = self.ax.transData.transform(offsets)
            distances: np.ndarray[float] = np.hypot(pixels[:, 0] - event.x, pixels[:, 1] - event.y)
            i: int = int(np.argmin(distances))
            if (distances[i] < 10) and ((best is None) or (distances[i] < best[0])):
                best = (float(distances[i]), float(offsets[i, 0]))
        if best is not None:
            self.peakClicked.emit(best[1])


class FigureTab(QtWidgets.QWidget):
    """Matplotlib figure with toolbar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure: Figure = Figure(layout='constrained', dpi=SCREEN_DPI)
        self.canvas: FigureCanvasQTAgg = FigureCanvasQTAgg(self.figure)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))
        layout.addWidget(self.canvas)

    def message(self, text: str) -> None:
        self.figure.clear()
        self.figure.text(.5, .5, text, ha='center', va='center', wrap=True)
        self.canvas.draw_idle()


class SpectraGui(QtWidgets.QMainWindow):
    """
    Window for selecting the parameters of the processing steps of a Spectra object.

    Steps only run when their button is clicked. Once a step on the summed
    spectrum has run (noise level, peaks, filtering of binned intensities),
    changing its parameters applies them right away. Results of later steps
    depending on it are reset, which is shown in their status (the GUI asks
    before discarding kernels or binned intensities).
    """

    def __init__(
            self,
            spectra: Spectra | None = None,
            reader: Any = None,
            n_jobs: int = 8,
            path_temp: str | None = None,
            path_folder: str | None = None,
            parent: QtWidgets.QWidget | None = None
    ) -> None:
        """
        Parameters
        ----------
        spectra: Spectra, optional
            The spectra to process (results are stored in it). The default is to
            open a measurement in the window.
        reader: RtmsBrukerMCFReader | Hdf5Handler | SpectraCache, optional
            Reader from which to obtain the spectra (required with spectra).
        n_jobs: int, optional
            Default number of threads for steps reading all spectra.
        path_temp: str, optional
            Default folder on a fast disk for a temporary copy of the spectra
            (see Spectra.set_temp_dir). The default is to not copy spectra.
        path_folder: str, optional
            Default folder of the measurement to open.
        """
        super().__init__(parent)
        assert (spectra is None) or (reader is not None), 'provide the reader of the spectra'
        self.spectra: Spectra | None = None
        self.reader: Any = None
        self.worker: Worker | None = None
        self._temp_dir_set: str | None = None
        self._reset_state()

        self.setWindowTitle('Spectra')
        self.resize(1500, 950)
        self._build(n_jobs, path_temp)
        self.folder_edit.setText(path_folder or '')
        if spectra is not None:
            self.set_spectra(spectra, reader)
        else:
            self.refresh()

        self._log_handler = LogHandler()
        self._log_handler.signal.message.connect(lambda text: self.statusBar().showMessage(text, 15_000))
        logging.getLogger('maspim').addHandler(self._log_handler)
        # PyQt aborts the program on exceptions in event handlers if no excepthook is set
        if sys.excepthook is sys.__excepthook__:
            sys.excepthook = self._excepthook

    def _excepthook(self, exc_type, exc_value, exc_traceback) -> None:
        """Show unexpected errors instead of closing the program."""
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        message: str = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        QtWidgets.QMessageBox.critical(self, 'Unexpected error', message[-3000:])

    def _reset_state(self) -> None:
        """Forget everything about the current measurement."""
        # summed spectrum before subtracting the baseline and whether it is calibrated
        self.raw_intensities: np.ndarray[float] | None = None
        self.sum_is_calibrated: bool = False
        # last uncalibrated sum, to compare it with the calibrated one
        self.uncalibrated_sum: np.ndarray[float] | None = None
        # per-peak attributes right after finding peaks, kernel fits start from them
        self.peaks_state: dict[str, Any] | None = None
        # per-peak attributes right after binning, filtering of binned intensities starts from them
        self.binned_state: dict[str, Any] | None = None
        # parameters with which each step was run last (None: unknown, e.g. loaded from file)
        self.applied: dict[str, dict[str, Any] | None] = {}
        # why the results of a step were reset
        self.reset_notes: dict[str, str] = {}
        # numbers of peaks before the kernel fit, after the fit and after filtering by sigma
        self.kernel_counts: tuple[int, int, int] | None = None
        # spectra for the calibration preview: (indices, intensities on spectra.mzs)
        self.preview: tuple[np.ndarray[int], list[np.ndarray[float]]] | None = None
        # mz of the peak clicked in the summed spectrum
        self.selected_mz: float | None = None
        # peaks used for the sigma estimate (positions in the peaks of the peak step)
        self.prominent: np.ndarray[int] | None = None
        # kernel parameters for which the reconstruction was computed, and the reconstruction
        self._reconstruction: tuple[np.ndarray[float] | None, np.ndarray[float] | None] = (None, None)

    # ------------------------------------------------------------------ layout
    def _build(self, n_jobs: int, path_temp: str | None) -> None:
        self.plot = SpectrumPlot()
        self.plot.peakClicked.connect(self.select_peak)
        self.calibration_tab = FigureTab()
        self.binned_tab = FigureTab()
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self.plot, 'Summed spectrum')
        tabs.addTab(self.calibration_tab, 'Calibration')
        tabs.addTab(self.binned_tab, 'Binned intensities')
        self.tabs = tabs

        # status of each step and which results a change of its parameters resets
        self.status_labels: dict[str, QtWidgets.QLabel] = {}
        self.reset_hints: dict[str, QtWidgets.QLabel] = {}
        controls = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(controls)
        layout.addWidget(self._build_measurement())
        # steps need a measurement
        self.step_groups: list[QtWidgets.QGroupBox] = [
            build() for build in (self._build_read, self._build_noise, self._build_calibration,
                                  self._build_peaks, self._build_kernels, self._build_binning, self._build_results)
        ]
        for group in self.step_groups:
            layout.addWidget(group)
        layout.addStretch()
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(controls)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(400)
        self.controls = controls

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(tabs)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        self.cancel_button = QtWidgets.QPushButton('Cancel')
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self._cancel)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_button)

        self.n_jobs.setValue(n_jobs)
        self.temp_edit.setText(path_temp or '')

        # debounce live updates of steps on the summed spectrum
        self._pending: list[Callable[[], None]] = []
        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._run_pending)
        self._live: dict[str, Callable[[], None]] = {
            stage: (lambda stage=stage: self.change(stage)) for stage in ('noise', 'peaks', 'filter')
        }

    def _group(self, title: str, stage: str | None = None) -> tuple[QtWidgets.QGroupBox, QtWidgets.QFormLayout]:
        group = QtWidgets.QGroupBox(title)
        form = QtWidgets.QFormLayout(group)
        if stage is not None:
            status = QtWidgets.QLabel()
            status.setWordWrap(True)
            self.status_labels[stage] = status
            form.addRow(status)
        return group, form

    def _add_reset_hint(self, stage: str, form: QtWidgets.QFormLayout) -> None:
        hint = QtWidgets.QLabel()
        hint.setWordWrap(True)
        hint.setStyleSheet(f'color: {COLORS["changed"]}; font-style: italic')
        self.reset_hints[stage] = hint
        form.addRow(hint)

    def _connect(self, stage: str, *signals: Any) -> None:
        for signal in signals:
            signal.connect(lambda *_, stage=stage: self._on_parameter_changed(stage))

    def _build_measurement(self) -> QtWidgets.QGroupBox:
        group, form = self._group('0. Measurement')
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setPlaceholderText('d-folder or folder containing it')
        self.folder_edit.setToolTip('the d-folder of the measurement or a folder containing only one d-folder')
        browse = QtWidgets.QToolButton()
        browse.setText('...')
        browse.clicked.connect(self._browse_folder)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.folder_edit)
        row.addWidget(browse)
        form.addRow('folder', row)
        self.load_saved = _check(False, 'continue from saved spectra', 'load the spectra saved in the d-folder')
        form.addRow('', self.load_saved)
        open_button = QtWidgets.QPushButton('Open')
        open_button.clicked.connect(self.open_measurement)
        self.folder_edit.returnPressed.connect(self.open_measurement)
        form.addRow(open_button)
        self.measurement_label = QtWidgets.QLabel('no measurement opened')
        self.measurement_label.setWordWrap(True)
        form.addRow(self.measurement_label)
        return group

    def _build_read(self) -> QtWidgets.QGroupBox:
        group, form = self._group('1. Read and sum spectra', 'sum')
        self.n_jobs = _spin(8, 1, os.cpu_count() or 1, 'threads processing spectra while they are read')
        form.addRow('threads', self.n_jobs)

        self.temp_edit = QtWidgets.QLineEdit()
        self.temp_edit.setPlaceholderText('no copy')
        self.temp_edit.setToolTip(
            'folder on a fast disk, spectra are copied there while they are read for the first time\n'
            'and later steps read the copy (deleted when the window is closed)'
        )
        browse = QtWidgets.QToolButton()
        browse.setText('...')
        browse.clicked.connect(self._browse_temp)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.temp_edit)
        row.addWidget(browse)
        form.addRow('fast disk', row)
        self.copy_size_label = QtWidgets.QLabel()
        form.addRow('', self.copy_size_label)

        self.sum_calibrated = _check(False, 'apply calibration', 'add up calibrated spectra')
        form.addRow('', self.sum_calibrated)
        self.sum_button = QtWidgets.QPushButton('Sum spectra')
        self.sum_button.clicked.connect(self.run_sum)
        form.addRow(self.sum_button)
        self._add_reset_hint('sum', form)
        return group

    def _build_noise(self) -> QtWidgets.QGroupBox:
        group, form = self._group('2. Noise level and baseline', 'noise')
        self.noise_auto = _check(True, 'estimate window from peak widths')
        self.noise_window = _dspin(.01, 1e-4, .999, 1e-3, 4, 'window of the minimum filter in Da')
        self.noise_window.setEnabled(False)
        self.noise_auto.toggled.connect(lambda auto: self.noise_window.setEnabled(not auto))
        form.addRow(self.noise_auto)
        form.addRow('window (Da)', self.noise_window)
        self.noise_button = QtWidgets.QPushButton('Estimate noise level and subtract baseline')
        self.noise_button.setToolTip('estimate the noise level of the summed spectrum and subtract the baseline\n'
                                     '(noise level x number of spectra), peaks are found in the subtracted spectrum')
        self.noise_button.clicked.connect(lambda: self.change('noise'))
        form.addRow(self.noise_button)
        self._add_reset_hint('noise', form)
        self._connect('noise', self.noise_auto.toggled, self.noise_window.valueChanged)
        return group

    def _build_calibration(self) -> QtWidgets.QGroupBox:
        group, form = self._group('3. Calibration', 'calibration')
        self.calibrants = QtWidgets.QLineEdit()
        # calibrants in the mass window of the last opened measurement
        self._default_calibrants: str = ''
        self.calibrants.setToolTip('exact masses of the calibrants in Da, separated by commas')
        form.addRow('calibrants', self.calibrants)
        self.search_range = _dspin(5, .1, 100, .5, 2, 'calibrant peaks are searched within +/- this range')
        form.addRow('search range (mDa)', self.search_range)
        self.calib_snr = _dspin(4, 0, 1000, .5, 1, 'minimum SNR of calibrant peaks, 0 uses the minimum height')
        form.addRow('SNR threshold', self.calib_snr)
        self.min_height = FloatEdit(1e4, 'minimum intensity of calibrant peaks (if the SNR threshold is 0)')
        form.addRow('min height', self.min_height)
        self.max_degree = _spin(1, 0, 3, 'maximum degree of the calibration polynomial')
        form.addRow('max degree', self.max_degree)
        self.nearest = _check(False, 'nearest peak', 'take the nearest instead of the highest peak in the range')
        form.addRow('', self.nearest)

        self.n_preview = _spin(6, 1, 30, 'number of random spectra to preview')
        preview_button = QtWidgets.QPushButton('Preview random spectra')
        preview_button.clicked.connect(self.load_preview)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.n_preview)
        row.addWidget(preview_button)
        form.addRow(row)
        self.calibrate_button = QtWidgets.QPushButton('Calibrate and sum spectra')
        self.calibrate_button.setToolTip(
            'find calibration functions and add up the calibrated spectra in one pass, '
            'the calibrated sum replaces the summed spectrum'
        )
        self.calibrate_button.clicked.connect(self.run_calibration)
        form.addRow(self.calibrate_button)
        self.summary_button = QtWidgets.QPushButton('Show calibration summary')
        self.summary_button.clicked.connect(self.show_calibration_summary)
        form.addRow(self.summary_button)
        self.show_uncalibrated = _check(False, 'show uncalibrated sum', 'compare with the uncalibrated sum in the plot')
        self.show_uncalibrated.toggled.connect(lambda *_: self.refresh())
        form.addRow('', self.show_uncalibrated)
        self._add_reset_hint('calibration', form)
        for signal in (self.calibrants.editingFinished, self.search_range.valueChanged,
                       self.calib_snr.valueChanged, self.min_height.valueChanged,
                       self.max_degree.valueChanged, self.nearest.toggled):
            signal.connect(lambda *_: self._schedule(self.update_preview))
        self._connect('calibration', self.calibrants.editingFinished, self.search_range.valueChanged,
                      self.calib_snr.valueChanged, self.min_height.valueChanged,
                      self.max_degree.valueChanged, self.nearest.toggled)
        return group

    def _build_peaks(self) -> QtWidgets.QGroupBox:
        group, form = self._group('4. Peaks', 'peaks')
        self.prominence = FloatEdit(.1, 'required prominence, values < 1 are relative to the median intensity')
        form.addRow('prominence', self.prominence)
        self.peak_width = _spin(3, 1, 10_000, 'minimum width of peaks in data points')
        form.addRow('min width (points)', self.peak_width)
        self.peak_distance = _spin(0, 0, 100_000, 'minimum distance between peaks in data points (0: none)')
        form.addRow('min distance (points)', self.peak_distance)
        self.peaks_snr = _dspin(0, 0, 10_000, 1, 1, 'remove peaks with a lower SNR')
        form.addRow('SNR threshold', self.peaks_snr)
        self.remove_side_peaks = _check(False, 'remove side peaks', 'remove peaks likely caused by spectral leakage')
        form.addRow('', self.remove_side_peaks)
        self.peaks_button = QtWidgets.QPushButton('Find peaks')
        self.peaks_button.clicked.connect(lambda: self.change('peaks'))
        form.addRow(self.peaks_button)
        self._add_reset_hint('peaks', form)
        self._connect('peaks', self.prominence.valueChanged, self.peak_width.valueChanged,
                      self.peak_distance.valueChanged, self.peaks_snr.valueChanged, self.remove_side_peaks.toggled)
        return group

    def _build_kernels(self) -> QtWidgets.QGroupBox:
        group, form = self._group('5. Kernels', 'kernels')
        self.bigaussian = _check(False, 'bigaussian kernels')
        self.fine_tune = _check(True, 'fine tune', 'fit kernels to the summed spectrum')
        self.discard_invalid = _check(True, 'discard invalid', 'discard peaks for which the fit failed')
        for check in (self.bigaussian, self.fine_tune, self.discard_invalid):
            form.addRow('', check)
        self.fit_sigma_max = _dspin(5, .01, 100, .5, 2, 'largest sigma of a fitted kernel (larger ones are halved)')
        form.addRow('max sigma in fit (mDa)', self.fit_sigma_max)
        self.sigma_filter = _check(False, 'remove kernels with sigma outside')
        self.sigma_filter.setToolTip(
            'remove kernels whose sigma is not strictly within the range, an upper limit equal to\n'
            'the max sigma of the fit removes kernels that reached it'
        )
        self.sigma_min = _dspin(0, 0, 100, .1, 2, 'smallest sigma of kept kernels')
        self.sigma_max = _dspin(5, 0, 100, .1, 2, 'largest sigma of kept kernels')
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.sigma_min)
        row.addWidget(QtWidgets.QLabel('to'))
        row.addWidget(self.sigma_max)
        row.addWidget(QtWidgets.QLabel('mDa'))
        form.addRow(self.sigma_filter)
        form.addRow('', row)
        self.n_prominent = _spin(20, 3, 1000, 'number of most prominent peaks for the estimate')
        estimate_button = QtWidgets.QPushButton('Estimate from prominent peaks')
        estimate_button.setToolTip(
            'estimate sigma of the most prominent peaks from their width at half maximum,\n'
            'sets the max sigma of the fit and the range of kept kernels'
        )
        estimate_button.clicked.connect(self.estimate_sigma)
        self.estimate_button = estimate_button
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.n_prominent)
        row.addWidget(estimate_button)
        form.addRow(row)
        self.sigma_label = QtWidgets.QLabel('click a peak in the summed spectrum to see its width')
        self.sigma_label.setWordWrap(True)
        form.addRow(self.sigma_label)
        self.kernels_button = QtWidgets.QPushButton('Fit kernels')
        self.kernels_button.setToolTip('fit kernels to the peaks found in step 4 (removed peaks come back)')
        self.kernels_button.clicked.connect(self.run_kernels)
        form.addRow(self.kernels_button)
        self._add_reset_hint('kernels', form)
        self._connect('kernels', self.bigaussian.toggled, self.fine_tune.toggled, self.discard_invalid.toggled,
                      self.fit_sigma_max.valueChanged, self.sigma_filter.toggled, self.sigma_min.valueChanged,
                      self.sigma_max.valueChanged)
        return group

    def _build_binning(self) -> QtWidgets.QGroupBox:
        group, form = self._group('6. Binning', 'binning')
        self.bin_method = QtWidgets.QComboBox()
        self.bin_method.addItems(['height', 'area', 'max'])
        self.bin_method.setToolTip(
            'height: intensity at the kernel center\narea: overlap with the kernel\n'
            'max: highest local maximum within one sigma (gaussian kernels only)'
        )
        form.addRow('method', self.bin_method)
        self.bin_button = QtWidgets.QPushButton('Bin spectra')
        self.bin_button.clicked.connect(self.run_binning)
        form.addRow(self.bin_button)
        form.addRow(QtWidgets.QLabel('filter binned intensities (applied right away):'))
        self.binned_snr = _dspin(0, 0, 10_000, .5, 1, 'set binned intensities with a lower SNR to 0')
        form.addRow('SNR threshold', self.binned_snr)
        self.intensity_min = FloatEdit(0, 'set binned intensities below this value to 0')
        form.addRow('min intensity', self.intensity_min)
        self.success_rate = _dspin(0, 0, 1, .01, 3, 'remove compounds found in a smaller fraction of spectra')
        form.addRow('min success rate', self.success_rate)
        self._connect('binning', self.bin_method.currentTextChanged)
        self._connect('filter', self.binned_snr.valueChanged, self.intensity_min.valueChanged,
                      self.success_rate.valueChanged)
        return group

    def _build_results(self) -> QtWidgets.QGroupBox:
        group, form = self._group('7. Results')
        self.feature_table_button = QtWidgets.QPushButton('Set feature table and save')
        self.feature_table_button.clicked.connect(self.save)
        form.addRow(self.feature_table_button)
        code_button = QtWidgets.QPushButton('Show parameters as code')
        code_button.clicked.connect(self.show_code)
        form.addRow(code_button)
        return group

    # ------------------------------------------------------------------ parameters
    def _params(self, stage: str) -> dict[str, Any]:
        """Current parameters of a step."""
        if stage == 'sum':
            return {}
        if stage == 'noise':
            return dict(auto=self.noise_auto.isChecked(), window=self.noise_window.value())
        if stage == 'calibration':
            return self._calibration_kwargs()
        if stage == 'peaks':
            return dict(prominence=self.prominence.value(), width=self.peak_width.value(),
                        distance=self.peak_distance.value(), snr=self.peaks_snr.value(),
                        side_peaks=self.remove_side_peaks.isChecked())
        if stage == 'kernels':
            return dict(bigaussian=self.bigaussian.isChecked(), fine_tune=self.fine_tune.isChecked(),
                        discard_invalid=self.discard_invalid.isChecked(), sigma_max=self.fit_sigma_max.value(),
                        thr_sigma=self._thr_sigma())
        if stage == 'binning':
            return dict(method=self.bin_method.currentText())
        if stage == 'filter':
            return dict(snr=self.binned_snr.value(), intensity_min=self.intensity_min.value(),
                        success_rate=self.success_rate.value())
        raise KeyError(stage)

    def _set_params(self, stage: str, params: dict[str, Any] | None) -> None:
        """Set the widgets of a step (without triggering updates)."""
        if params is None:
            return
        widgets: dict[str, QtWidgets.QWidget] = {
            'noise': dict(auto=self.noise_auto, window=self.noise_window),
            'peaks': dict(prominence=self.prominence, width=self.peak_width, distance=self.peak_distance,
                          snr=self.peaks_snr, side_peaks=self.remove_side_peaks),
            'filter': dict(snr=self.binned_snr, intensity_min=self.intensity_min, success_rate=self.success_rate),
        }.get(stage, {})
        for name, widget in widgets.items():
            widget.blockSignals(True)
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(params[name])
            elif isinstance(widget, FloatEdit):
                widget._value = params[name]
                widget.setText(f'{params[name]:g}')
            else:
                widget.setValue(params[name])
            widget.blockSignals(False)
        if stage == 'noise':
            self.noise_window.setEnabled(not self.noise_auto.isChecked())

    def _calibration_kwargs(self) -> dict[str, Any]:
        return dict(
            calibrants_mz=[float(mz) for mz in self.calibrants.text().replace(';', ',').split(',') if mz.strip()],
            search_range=self.search_range.value() * 1e-3,
            calib_snr_threshold=self.calib_snr.value(),
            min_height=self.min_height.value(),
            max_degree=self.max_degree.value(),
            nearest=self.nearest.isChecked()
        )

    def _thr_sigma(self) -> tuple[float, float] | None:
        if not self.sigma_filter.isChecked():
            return None
        return self.sigma_min.value() * 1e-3, self.sigma_max.value() * 1e-3

    # ------------------------------------------------------------------ measurement
    def _browse_folder(self) -> None:
        folder: str = QtWidgets.QFileDialog.getExistingDirectory(self, 'Measurement', self.folder_edit.text())
        if folder:
            self.folder_edit.setText(folder)

    def open_measurement(self) -> None:
        """Open the measurement in the folder field in the background."""
        try:
            path_d_folder: str = find_d_folder(self.folder_edit.text().strip())
        except (FileNotFoundError, NotADirectoryError) as e:
            QtWidgets.QMessageBox.warning(self, 'Open', str(e))
            return
        load_saved: bool = self.load_saved.isChecked()

        def job() -> tuple[Spectra, Any]:
            reader = RtmsBrukerMCFReader.from_dir(path_d_folder)
            if load_saved:
                spectra: Spectra = Spectra.load_from_file(path_folder=path_d_folder)
            else:
                spectra: Spectra = Spectra.from_reader(reader)
            return spectra, reader

        self._start(job, lambda result: self.set_spectra(*result), f'opening {path_d_folder}')

    def set_spectra(self, spectra: Spectra, reader: Any) -> None:
        """Process spectra (read with reader) in the window, replacing the previous ones."""
        if (self.spectra is not None) and (self._temp_dir_set is not None):
            self.spectra.remove_temp_dir()
        self._temp_dir_set = None
        self.spectra, self.reader = spectra, reader
        self._reset_state()

        name: str = os.path.basename(os.path.normpath(str(getattr(spectra, 'path_folder', '') or '')))
        self.setWindowTitle(f'Spectra: {name}')
        self.measurement_label.setText(
            f'{name}: {spectra._n_spectra} spectra, {len(spectra.mzs)} mz values '
            f'in {spectra.limits[0]:g} to {spectra.limits[1]:g} Da'
        )
        n_bytes: int = 4 * spectra._n_spectra * len(spectra.mzs)
        size: str = f'{n_bytes / 1024 ** 3:.1f} GB' if n_bytes > 1024 ** 3 else f'{n_bytes / 1024 ** 2:.0f} MB'
        self.copy_size_label.setText(f'copy needs about {size}')
        # keep calibrants entered by hand
        if self.calibrants.text() in ('', self._default_calibrants):
            self._default_calibrants = ', '.join(f'{mz:.5f}' for mz in get_calibrants(spectra.limits))
            self.calibrants.setText(self._default_calibrants)
        self.calibration_tab.message('preview random spectra to check the calibration parameters')
        self.binned_tab.message('bin spectra to see the binned intensities')

        # show results already stored in the spectra (e.g. loaded from disk), parameters are unknown
        s: Spectra = spectra
        if check_attr(s, '_intensities') and np.any(s._intensities):
            self.raw_intensities = s._intensities.copy()
            if getattr(s, '_noise_level_subtracted', False) and check_attr(s, '_noise_level'):
                self.raw_intensities += s._noise_level * s._n_spectra
            self.sum_is_calibrated = check_attr(s, '_calibration_parameters')
        for stage in ('sum', 'noise', 'calibration', 'peaks', 'kernels', 'binning'):
            if self._has(stage):
                self.applied[stage] = None
        if self._has('peaks'):
            # with kernels, these are only the peaks that have kernels
            self._store_peaks_state()
        if self._has('binning'):
            self._store_binned_state()
            self.applied['filter'] = None
        self.refresh()
        self.plot.reset_view()

    # ------------------------------------------------------------------ state of the steps
    def _has(self, stage: str) -> bool:
        """Whether the results of a step exist."""
        s: Spectra | None = self.spectra
        if s is None:
            return False
        if stage == 'sum':
            return self.raw_intensities is not None
        if stage == 'noise':
            return check_attr(s, '_noise_level')
        if stage == 'calibration':
            return check_attr(s, '_calibration_parameters')
        if stage == 'peaks':
            return check_attr(s, '_peaks')
        if stage == 'kernels':
            return check_attr(s, '_kernel_params')
        if stage in ('binning', 'filter'):
            return check_attr(s, '_line_spectra')
        raise KeyError(stage)

    def _results_after(self, stage: str) -> list[str]:
        """Existing results of the steps depending on stage (in order)."""
        # calibrating replaces the summed spectrum
        position: int = STAGES.index('sum' if stage == 'calibration' else stage)
        return [later for later in STAGES[position + 1:] if self._has(later)]

    def _confirm_reset(self, lost: list[str], action: str) -> bool:
        """Ask before results that take long to recompute are discarded."""
        if not any(stage in EXPENSIVE for stage in lost):
            return True
        answer = QtWidgets.QMessageBox.question(
            self, 'Discard results?',
            f'{action} discards the {_join(STAGE_LABELS[stage] for stage in lost)}, which have to be '
            f'computed again. Continue?'
        )
        return answer == QtWidgets.QMessageBox.Yes

    def _note_reset(self, lost: list[str], reason: str) -> None:
        for stage in lost:
            self.reset_notes[stage] = reason
            self.applied.pop(stage, None)
        if 'binning' in lost:
            self.applied.pop('filter', None)
        if lost:
            self.statusBar().showMessage(f'{reason}: reset the {_join(STAGE_LABELS[stage] for stage in lost)}', 15_000)

    def _on_parameter_changed(self, stage: str) -> None:
        """Apply parameters of steps on the summed spectrum right away, if they were run already."""
        if (stage in self._live) and (stage in self.applied):
            self._schedule(self._live[stage])
        else:
            self.refresh()

    def change(self, stage: str) -> None:
        """Run a step on the summed spectrum (noise, peaks, filter) with the current parameters."""
        if (self.spectra is None) or (self.worker is not None):
            return
        lost: list[str] = self._results_after(stage) if stage != 'filter' else []
        label: str = STAGE_LABELS[stage]
        if not self._confirm_reset(lost, f'Changing the {label}'):
            self._set_params(stage, self.applied.get(stage))
            self.refresh()
            return
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            {'noise': self._apply_noise, 'peaks': self._apply_peaks, 'filter': self._apply_filter}[stage]()
        except Exception as e:
            logger.exception(e)
            QtWidgets.QMessageBox.critical(self, 'Error', f'{type(e).__name__}: {e}')
        else:
            self.applied[stage] = self._params(stage)
            self.reset_notes.pop(stage, None)
            self._note_reset(lost, f'{label} changed')
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.refresh()

    def _apply_noise(self) -> None:
        if self.raw_intensities is None:
            return
        s: Spectra = self.spectra
        s._intensities = self.raw_intensities.copy()
        # also resets peaks, kernels and binned intensities
        s.reset_noise_level()
        s.set_noise_level(window_size=None if self.noise_auto.isChecked() else self.noise_window.value())
        s.subtract_baseline()
        self.peaks_state = self.binned_state = None
        self.prominent = None

    def _apply_peaks(self) -> None:
        s: Spectra = self.spectra
        if not self._has('noise'):
            raise ValueError('estimate the noise level first')
        s.set_peaks(
            prominence=self.prominence.value(), width=self.peak_width.value(),
            distance=self.peak_distance.value() or None
        )
        if (self.peaks_snr.value() > 0) or self.remove_side_peaks.isChecked():
            s.filter_peaks(peaks_snr_threshold=self.peaks_snr.value(), remove_sidepeaks=self.remove_side_peaks.isChecked())
        self._store_peaks_state()
        self.binned_state = None
        self.prominent = None
        self.kernel_counts = None

    def _apply_filter(self) -> None:
        """Filter the binned intensities, starting from the state right after binning."""
        if self.binned_state is None:
            return
        s: Spectra = self.spectra
        for name, value in self.binned_state.items():
            setattr(s, name, copy.deepcopy(value))
        s.filter_line_spectra(
            binned_snr_threshold=self.binned_snr.value(), intensity_min=self.intensity_min.value(),
            success_rate_threshold=self.success_rate.value()
        )

    def _store_peaks_state(self) -> None:
        self.peaks_state = {name: copy.deepcopy(getattr(self.spectra, name)) for name in PEAK_STEP_ATTRS}

    def _store_binned_state(self) -> None:
        self.binned_state = {name: copy.deepcopy(getattr(self.spectra, name)) for name in PEAK_ATTRS}

    def _schedule(self, update: Callable[[], None]) -> None:
        if update not in self._pending:
            self._pending.append(update)
        self._debounce.start()

    def _run_pending(self) -> None:
        if self.worker is not None:
            # never change the spectra while a background step uses them
            self._debounce.start()
            return
        pending, self._pending = self._pending, []
        for update in pending:
            try:
                update()
            except Exception as e:
                logger.exception(e)
                self.statusBar().showMessage(f'{type(e).__name__}: {e}', 15_000)

    # ------------------------------------------------------------------ display
    def _baseline(self) -> np.ndarray[float] | float:
        s: Spectra = self.spectra
        if getattr(s, '_noise_level_subtracted', False) and check_attr(s, '_noise_level'):
            return s._noise_level * s._n_spectra
        return 0.

    def _status(self, stage: str) -> tuple[str, str]:
        """State (key of COLORS) and description of a step."""
        s: Spectra = self.spectra
        note: str | None = self.reset_notes.get(stage)
        changed: bool = (stage in self.applied) and (self.applied[stage] is not None) \
            and (self.applied[stage] != self._params(stage))
        if not self._has(stage):
            if note is not None:
                return 'changed', f'reset ({note}), run again'
            requirement: str | None = {
                'noise': None if self._has('sum') else 'sum spectra first',
                'calibration': None if self._has('sum') else 'sum spectra first (for the noise level)',
                'peaks': None if self._has('noise') else 'estimate the noise level first',
                'kernels': None if self._has('peaks') else 'find peaks first',
                'binning': None if self._has('kernels') else 'fit kernels first',
            }.get(stage)
            return 'not run', 'not run' + (f' ({requirement})' if requirement else '')
        if stage == 'sum':
            text = f'done: {"calibrated" if self.sum_is_calibrated else "uncalibrated"} sum of {s._n_spectra} spectra'
        elif stage == 'noise':
            window_points: int = int(s._noise_level_parameters['window_size']) \
                if check_attr(s, '_noise_level_parameters') else 0
            dmz: float = float(np.median(np.diff(s.mzs)))
            text = (f'done: window {window_points} points (~{window_points * dmz * 1e3:.1f} mDa), '
                    f'baseline {"subtracted" if s._noise_level_subtracted else "not subtracted"}')
        elif stage == 'calibration':
            calibrated: int = int(np.any(s._calibration_parameters, axis=1).sum())
            text = f'done: {calibrated} of {len(s._calibration_parameters)} spectra calibrated'
            if changed:
                return 'changed', text + '; parameters changed since, calibrate again to apply them'
        elif stage == 'peaks':
            text = f'done: {len(s._peaks)} peaks'
        elif stage == 'kernels':
            loss: float = np.sum(np.abs(s.intensities - self._reconstructed())) / max(np.sum(s.intensities), 1e-30)
            text = f'done: {len(s.kernel_params)} {s._kernel_shape} kernels, loss {loss:.1%}'
            if self.kernel_counts is not None:
                n_peaks, n_fit, n_kept = self.kernel_counts
                removed: list[str] = []
                if n_fit < n_peaks:
                    removed.append(f'{n_peaks - n_fit} failed fits')
                if n_kept < n_fit:
                    removed.append(f'{n_fit - n_kept} outside the sigma range')
                if removed:
                    text += f' ({n_peaks} peaks, removed {", ".join(removed)})'
            if changed:
                return 'changed', text + '; parameters changed since, fit again to apply them'
        else:  # binning
            success: np.ndarray[float] = (s._line_spectra > 0).mean(axis=0)
            text = (f'done: {s._line_spectra.shape[1]} compounds binned by {s._binning_by}, '
                    f'median success rate {np.median(success):.0%}')
            if changed:
                return 'changed', text + '; method changed since, bin again to apply it'
        if (stage in self.applied) and (self.applied[stage] is None):
            text += ' (loaded, parameters unknown)'
        return 'done', text

    def refresh(self) -> None:
        """Update plot, status and buttons from the state of the spectra."""
        s: Spectra | None = self.spectra
        self.controls.setEnabled(self.worker is None)
        for group in self.step_groups:
            group.setEnabled(s is not None)
        if s is None:
            self.plot.clear()
            self.plot.update_plot()
            return

        for stage, label in self.status_labels.items():
            state, text = self._status(stage)
            label.setText(text)
            label.setStyleSheet(f'color: {COLORS[state]}; font-weight: bold')
        for stage, hint in self.reset_hints.items():
            lost: list[str] = self._results_after(stage)
            action: str = {'sum': 'summing again', 'calibration': 'calibrating again',
                           'kernels': 'fitting again'}.get(stage, 'changing these parameters')
            hint.setText(f'{action} resets the {_join(STAGE_LABELS[later] for later in lost)}' if lost else '')
            hint.setVisible(bool(lost))

        raw: np.ndarray[float] | None = self.raw_intensities
        has_sum: bool = raw is not None
        has_peaks: bool = self._has('peaks') and (len(s._peaks) > 0)
        has_kernels: bool = self._has('kernels')
        has_binned: bool = self._has('binning')
        baseline = self._baseline()

        self.plot.set_curve('spectrum', s.mzs if has_sum else None, raw)
        show_uncalibrated: bool = (self.show_uncalibrated.isChecked() and self.sum_is_calibrated
                                   and (self.uncalibrated_sum is not None))
        self.plot.set_curve('uncalibrated', s.mzs if show_uncalibrated else None, self.uncalibrated_sum)
        self.plot.set_curve(
            'baseline', s.mzs if (has_sum and self._has('noise')) else None,
            np.broadcast_to(baseline, s.mzs.shape) if self._has('noise') else None
        )
        self.plot.set_markers('peaks', *((s.mzs[s._peaks], raw[s._peaks]) if (has_sum and has_peaks) else (None,)))
        removed: np.ndarray[int] = self._removed_peaks()
        self.plot.set_markers('removed', *((s.mzs[removed], raw[removed]) if (has_sum and len(removed)) else (None,)))
        prominent: np.ndarray[int] | None = None
        if has_sum and (self.prominent is not None) and (self.peaks_state is not None):
            prominent = self.peaks_state['_peaks'][self.prominent]
        self.plot.set_markers('prominent', *((s.mzs[prominent], raw[prominent]) if prominent is not None else (None,)))
        selected: int | None = self._nearest_peak(self.selected_mz) if has_sum else None
        self.plot.set_markers('selected', *(([s.mzs[selected]], [raw[selected]]) if selected is not None else (None,)))
        self.plot.set_curve('reconstructed', *(
            (s.mzs, self._reconstructed() + baseline) if (has_sum and has_kernels) else (None,)
        ))
        if has_sum and has_binned and has_kernels:
            centers: np.ndarray[float] = s.kernel_params[:, 0]
            # spectra are binned with their noise, so the sum compares to the summed spectrum without
            # subtracting the baseline
            self.plot.set_binned(centers, np.zeros_like(centers), s._line_spectra.sum(axis=0))
            self._plot_binned()
        else:
            self.plot.set_binned(None)
        self.plot.update_plot()

        has_calibration: bool = self._has('calibration')
        self.sum_calibrated.setEnabled(has_calibration)
        if not has_calibration:
            self.sum_calibrated.setChecked(False)
        self.noise_button.setEnabled(has_sum)
        self.calibrate_button.setEnabled(has_sum)
        self.summary_button.setEnabled(has_calibration)
        self.show_uncalibrated.setEnabled(self.sum_is_calibrated and (self.uncalibrated_sum is not None))
        self.peaks_button.setEnabled(self._has('noise'))
        self.estimate_button.setEnabled(has_peaks)
        self.kernels_button.setEnabled(has_peaks)
        self.bin_button.setEnabled(has_kernels)
        self.feature_table_button.setEnabled(has_binned)

    def _removed_peaks(self) -> np.ndarray[int]:
        """Peaks of the peak step without kernel (removed by the kernel fit or filtering)."""
        s: Spectra = self.spectra
        if (self.peaks_state is None) or (not self._has('kernels')):
            return np.empty(0, dtype=int)
        return np.setdiff1d(self.peaks_state['_peaks'], s._peaks)

    def _nearest_peak(self, mz: float | None) -> int | None:
        """Index in spectra.mzs of the peak (of the peak step) closest to mz."""
        if mz is None:
            return None
        s: Spectra = self.spectra
        peaks: np.ndarray[int] | None = self.peaks_state['_peaks'] if self.peaks_state is not None else (
            s._peaks if self._has('peaks') else None
        )
        if (peaks is None) or (len(peaks) == 0):
            return None
        return int(peaks[np.argmin(np.abs(s.mzs[peaks] - mz))])

    def _reconstructed(self) -> np.ndarray[float]:
        """Summed spectrum described by the kernels (cached for the current kernel parameters)."""
        s: Spectra = self.spectra
        params, reconstruction = self._reconstruction
        if params is not s._kernel_params:
            kernels = s._get_kernels_sparse(norm_mode='height')
            reconstruction = kernels @ s.kernel_params[:, 1]
            self._reconstruction = (s._kernel_params, reconstruction)
        return reconstruction

    # ------------------------------------------------------------------ kernel widths
    def _sigma_estimates(self, positions: np.ndarray[int]) -> np.ndarray[float]:
        """Sigma (in Da) from the width at half maximum of peaks (positions in the peaks of the peak step)."""
        s: Spectra = self.spectra
        state: dict[str, Any] = {name: getattr(s, name) for name in ('_peaks', '_peak_properties')}
        try:
            s._peaks, s._peak_properties = self.peaks_state['_peaks'], self.peaks_state['_peak_properties']
            return np.array([s._gaussian_from_peak(int(i))[2] for i in positions])
        finally:
            s._peaks, s._peak_properties = state['_peaks'], state['_peak_properties']

    def estimate_sigma(self) -> None:
        """Set the sigma parameters of the kernels from the most prominent peaks."""
        if self.peaks_state is None:
            return
        prominences: np.ndarray[float] = self.peaks_state['_peak_properties']['prominences']
        self.prominent = np.argsort(prominences)[::-1][:self.n_prominent.value()]
        sigmas: np.ndarray[float] = self._sigma_estimates(self.prominent) * 1e3
        lower, upper, median = float(sigmas.min()), float(sigmas.max()), float(np.median(sigmas))
        for widget, value in ((self.fit_sigma_max, 2 * upper), (self.sigma_min, lower / 2),
                              (self.sigma_max, 2 * upper)):
            widget.setValue(round(value, widget.decimals()))
        self.sigma_filter.setChecked(True)
        self.sigma_label.setText(
            f'{len(sigmas)} most prominent peaks (circled): sigma {lower:.2f} to {upper:.2f} mDa '
            f'(median {median:.2f}), set the max sigma of the fit to twice the largest and the kept '
            f'range from half the smallest to twice the largest'
        )
        self.refresh()

    def select_peak(self, mz: float) -> None:
        """Show the width of the peak clicked in the summed spectrum."""
        s: Spectra = self.spectra
        if (s is None) or (self.peaks_state is None) or (self.worker is not None):
            return
        peak: int | None = self._nearest_peak(mz)
        self.selected_mz = float(s.mzs[peak])
        position: int = int(np.flatnonzero(self.peaks_state['_peaks'] == peak)[0])
        sigma: float = float(self._sigma_estimates(np.array([position]))[0]) * 1e3
        text: str = f'peak at {s.mzs[peak]:.4f} Da: sigma from half maximum {sigma:.2f} mDa'
        if self._has('kernels'):
            kernel: np.ndarray[int] = np.flatnonzero(s._peaks == peak)
            text += (f', kernel sigma {s.kernel_params[kernel[0], 2] * 1e3:.2f} mDa' if len(kernel)
                     else ', removed by the kernel fit')
        self.sigma_label.setText(text)
        self.refresh()

    # ------------------------------------------------------------------ calibration
    def load_preview(self) -> None:
        """Read random spectra for the calibration preview."""
        s: Spectra = self.spectra
        indices: np.ndarray[int] = np.sort(np.random.default_rng().choice(
            s.indices, size=min(self.n_preview.value(), s._n_spectra), replace=False
        ))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            reader = s._cached_reader(self.reader)
            intensities = [
                s.get_spectrum(reader, index, intensities_only=True, recalibrate=False, resample=True)[1]
                for index in indices
            ]
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.preview = (indices, intensities)
        self.update_preview()
        self.tabs.setCurrentWidget(self.calibration_tab)

    def _picked_calibrant_peak(
            self, intensities: np.ndarray[float], calibrant: float, kwargs: dict[str, Any],
            noise_level: np.ndarray[float] | None
    ) -> int | None:
        """Index of the peak used for the calibrant (as in find_polycalibration_spectrum)."""
        mzs: np.ndarray[float] = self.spectra.mzs
        threshold = noise_level * kwargs['calib_snr_threshold'] if noise_level is not None else kwargs['min_height']
        peaks: np.ndarray[int] = find_peaks(intensities, height=threshold)[0]
        peaks = peaks[np.abs(mzs[peaks] - calibrant) < kwargs['search_range']]
        if len(peaks) == 0:
            return None
        if kwargs['nearest']:
            return int(peaks[np.argmin(np.abs(mzs[peaks] - calibrant))])
        return int(peaks[np.argmax(intensities[peaks])])

    def update_preview(self) -> None:
        """Calibrate the preview spectra and show them around the calibrants."""
        if self.preview is None:
            return
        s: Spectra = self.spectra
        kwargs: dict[str, Any] = self._calibration_kwargs()
        calibrants: list[float] = kwargs['calibrants_mz']
        if len(calibrants) == 0:
            self.calibration_tab.message('no calibrants')
            return
        snr_threshold: float = kwargs['calib_snr_threshold']
        if (snr_threshold > 0) and (not self._has('noise')):
            self.calibration_tab.message('estimate the noise level or set the SNR threshold to 0')
            return
        noise_level: np.ndarray[float] | None = s._noise_level if self._has('noise') else None
        noise_used: np.ndarray[float] | None = noise_level if snr_threshold > 0 else None
        indices, spectra = self.preview

        results = [
            find_polycalibration_spectrum(mzs=s.mzs, intensities=y, noise_level=noise_used, **kwargs)
            for y in spectra
        ]
        figure: Figure = self.calibration_tab.figure
        figure.clear()
        n_shown: int = min(len(calibrants), 4)
        axs = figure.subplots(
            nrows=(n_shown + 1) // 2, ncols=min(n_shown, 2), squeeze=False
        ).ravel()[:n_shown]
        for ax in figure.axes[n_shown:]:  # odd number of calibrants
            ax.remove()
        search_range: float = kwargs['search_range']
        for ax, calibrant, jt in zip(axs, calibrants, range(n_shown)):
            window = (calibrant - 3 * search_range, calibrant + 3 * search_range)
            mask = (s.mzs >= window[0]) & (s.mzs <= window[1])
            ax.axvspan(calibrant - search_range, calibrant + search_range, color='grey', alpha=.15,
                       label='search range')
            ax.axvline(calibrant, color='k', linewidth=1, label='calibrant')
            # the threshold in use is drawn solid, the other one dotted
            ax.axhline(kwargs['min_height'], color='r' if snr_threshold == 0 else '0.5',
                       linestyle='-' if snr_threshold == 0 else ':', linewidth=1.2 if snr_threshold == 0 else 1,
                       label=f'min height{" (used)" if snr_threshold == 0 else ""}')
            if noise_level is not None:
                ax.plot(s.mzs[mask], noise_level[mask] * max(snr_threshold, 1),
                        color='r' if snr_threshold > 0 else '0.5', linestyle='-' if snr_threshold > 0 else ':',
                        linewidth=1.2 if snr_threshold > 0 else 1,
                        label=f'SNR threshold x noise{" (used)" if snr_threshold > 0 else " (at SNR 1)"}')
            for it, (index, y, (p, n_coeffs, presences)) in enumerate(zip(indices, spectra, results)):
                color: str = f'C{it % 10}'
                ax.plot(s.mzs[mask], y[mask], color=color, linewidth=.8, alpha=.4)
                picked: int | None = self._picked_calibrant_peak(y, calibrant, kwargs, noise_used)
                if picked is not None:
                    ax.plot(s.mzs[picked], y[picked], marker='v', color=color, markersize=7, alpha=.6)
                if n_coeffs > 0:
                    mzs_calibrated: np.ndarray[float] = s.mzs + np.polyval(p, s.mzs)
                    mask_calibrated = (mzs_calibrated >= window[0]) & (mzs_calibrated <= window[1])
                    ax.plot(mzs_calibrated[mask_calibrated], y[mask_calibrated], color=color, linewidth=1.2,
                            label=f'{index}: {np.polyval(p, calibrant) * 1e3:+.2f} mDa')
                else:
                    ax.plot([], [], color=color, label=f'{index}: not calibrated')
            n_found: int = sum(presences[jt] for _, _, presences in results)
            ax.set_title(f'{calibrant:.4f} Da, found in {n_found}/{len(spectra)}', fontsize='medium')
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        for ax in axs[-min(n_shown, 2):]:
            ax.set_xlabel('m/z in Da')
        for ax in axs[::2]:
            ax.set_ylabel('intensity')
        handles, labels = axs[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside right upper', fontsize='small',
                      title='spectrum: shift\n(faint: uncalibrated,\ntriangle: picked peak)')
        if len(calibrants) > n_shown:
            figure.suptitle(f'first {n_shown} of {len(calibrants)} calibrants', fontsize='medium')
        self.calibration_tab.canvas.draw_idle()

    def show_calibration_summary(self) -> None:
        """Statistics of the calibration and the summed spectrum before and after calibrating."""
        s: Spectra = self.spectra
        if not self._has('calibration'):
            return
        params: np.ndarray[float] = s._calibration_parameters
        settings: dict[str, Any] = s._calibration_settings or {}
        calibrants: list[float] = list(settings.get('calibrants', []))
        presences_rate: dict[float, float] = settings.get('presences calibrants', {})
        n_spectra: int = len(params)
        calibrated: np.ndarray[bool] = np.any(params, axis=1)

        figure: Figure = self.calibration_tab.figure
        figure.clear()
        n_windows: int = min(len(calibrants), 4)
        grid = figure.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.1, 2, .8])
        ax_text = figure.add_subplot(grid[0, 0])
        ax_shift = figure.add_subplot(grid[0, 1])
        ax_hist = figure.add_subplot(grid[0, 2], sharey=ax_shift)
        sub = grid[1, :].subgridspec(1, max(n_windows, 1))
        axs_windows = [figure.add_subplot(sub[0, i]) for i in range(n_windows)]

        # statistics
        lines: list[str] = [
            f'calibrated spectra: {calibrated.sum()} of {n_spectra} ({calibrated.mean():.1%})',
            f'not calibrated (no calibrant found): {n_spectra - calibrated.sum()}',
        ]
        if (s._calibrator_presences is not None) and (len(s._calibrator_presences) == n_spectra):
            counts: np.ndarray[int] = np.bincount(s._calibrator_presences.sum(axis=1),
                                                  minlength=len(calibrants) + 1)
            lines.append('calibrants per spectrum: ' + ', '.join(
                f'{n}: {c}' for n, c in enumerate(counts) if c > 0))
        degrees: np.ndarray[int] = params.shape[1] - 1 - np.argmax(params != 0, axis=1)
        lines.append('polynomial degree: ' + ', '.join(
            f'{d}: {c}' for d, c in zip(*np.unique(degrees[calibrated], return_counts=True))))
        lines.append('shift at calibrant, median [5 %, 95 %]:')
        for calibrant in calibrants:
            shifts: np.ndarray[float] = np.polyval(params.T, calibrant)[calibrated] * 1e3
            if len(shifts) == 0:
                continue
            q05, q50, q95 = np.percentile(shifts, [5, 50, 95])
            lines.append(f'  {calibrant:.4f}: {q50:+.2f} [{q05:+.2f}, {q95:+.2f}] mDa, '
                         f'found in {presences_rate.get(calibrant, np.nan):.0%}')
        comparison: bool = (self.uncalibrated_sum is not None) and self.sum_is_calibrated
        if comparison:
            before, after = self._fwhm_of_prominent_peaks()
            if len(before):
                ratio: float = float(np.median(after / before))
                lines.append(f'FWHM of the {len(before)} most prominent peaks: median '
                             f'{np.median(before) * 1e3:.2f} -> {np.median(after) * 1e3:.2f} mDa ({ratio - 1:+.0%})')
        else:
            lines.append('sum spectra without calibration first to compare the peak widths')
        ax_text.axis('off')
        ax_text.text(0, 1, '\n'.join(lines), va='top', ha='left', fontsize='small', family='monospace',
                     transform=ax_text.transAxes, wrap=True)

        # shifts per spectrum
        for jt, calibrant in enumerate(calibrants[:4]):
            shifts = np.polyval(params.T, calibrant) * 1e3
            ax_shift.scatter(np.flatnonzero(calibrated), shifts[calibrated], s=2, color=f'C{jt}',
                             label=f'{calibrant:.4f} Da')
            ax_hist.hist(shifts[calibrated], bins=50, orientation='horizontal', color=f'C{jt}', alpha=.5)
        ax_shift.set_xlabel('spectrum')
        ax_shift.set_ylabel('shift at calibrant in mDa')
        ax_shift.legend(fontsize='small', markerscale=5)
        ax_hist.set_xlabel('spectra')
        ax_hist.tick_params(labelleft=False)

        # summed spectrum before and after calibrating around the calibrants
        search_range: float = settings.get('search_range', 5e-3)
        for ax, calibrant in zip(axs_windows, calibrants):
            window = (calibrant - 4 * search_range, calibrant + 4 * search_range)
            mask = (s.mzs >= window[0]) & (s.mzs <= window[1])
            title: str = f'{calibrant:.4f} Da'
            if comparison:
                ax.plot(s.mzs[mask], self.uncalibrated_sum[mask], color='0.5', linewidth=1, label='uncalibrated')
                widths = [fwhm_near(s.mzs, y, calibrant, 2 * search_range)
                          for y in (self.uncalibrated_sum, self.raw_intensities)]
                if None not in widths:
                    title += f'\nFWHM {widths[0][1] * 1e3:.2f} -> {widths[1][1] * 1e3:.2f} mDa'
            ax.plot(s.mzs[mask], self.raw_intensities[mask], color='C0', linewidth=1.2, label='calibrated')
            ax.axvline(calibrant, color='k', linewidth=1)
            ax.set_title(title, fontsize='small')
            ax.set_xlabel('m/z in Da')
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        if axs_windows:
            axs_windows[0].set_ylabel('summed intensity')
            axs_windows[0].legend(fontsize='small')
        self.calibration_tab.canvas.draw_idle()
        self.tabs.setCurrentWidget(self.calibration_tab)

    def _fwhm_of_prominent_peaks(self, n: int = 50) -> tuple[np.ndarray[float], np.ndarray[float]]:
        """FWHM (in Da) of the most prominent peaks of the calibrated sum, in the uncalibrated and calibrated sum."""
        s: Spectra = self.spectra
        calibrated: np.ndarray[float] = self.raw_intensities
        peaks, properties = find_peaks(calibrated, prominence=0)
        top: np.ndarray[int] = peaks[np.argsort(properties['prominences'])[::-1][:n]]
        before, after = [], []
        for peak in top:
            # search the peak in a window of a few widths
            half_window: float = 20 * float(np.diff(s.mzs[max(peak - 1, 0):peak + 1]).max(initial=1e-4))
            width_after = fwhm_near(s.mzs, calibrated, float(s.mzs[peak]), half_window)
            width_before = fwhm_near(s.mzs, self.uncalibrated_sum, float(s.mzs[peak]), half_window)
            if (width_after is not None) and (width_before is not None):
                before.append(width_before[1])
                after.append(width_after[1])
        return np.array(before), np.array(after)

    def _plot_binned(self) -> None:
        s: Spectra = self.spectra
        figure: Figure = self.binned_tab.figure
        figure.clear()
        ax = figure.subplots()
        success: np.ndarray[float] = (s._line_spectra > 0).mean(axis=0)
        mean: np.ndarray[float] = s._line_spectra.sum(axis=0) / np.maximum((s._line_spectra > 0).sum(axis=0), 1)
        points = ax.scatter(s.kernel_params[:, 0], success, c=np.log10(np.maximum(mean, 1)), s=8, cmap='viridis')
        figure.colorbar(points, ax=ax, label='log10 mean intensity (where found)')
        ax.set_xlabel('m/z in Da')
        ax.set_ylabel('fraction of spectra with intensity > 0')
        ax.set_ylim(0, 1.02)
        self.binned_tab.canvas.draw_idle()

    # ------------------------------------------------------------------ steps reading all spectra
    def _browse_temp(self) -> None:
        folder: str = QtWidgets.QFileDialog.getExistingDirectory(self, 'Folder on a fast disk', self.temp_edit.text())
        if folder:
            self.temp_edit.setText(folder)

    def _apply_temp_dir(self) -> None:
        path: str | None = self.temp_edit.text().strip() or None
        if path != self._temp_dir_set:
            self.spectra.set_temp_dir(path)
            self._temp_dir_set = path

    def _start(self, func: Callable[[], Any], on_done: Callable[[Any], None], message: str) -> None:
        """Run func in the background, on_done is called with its result."""
        # apply parameter changes made right before
        self._debounce.stop()
        self._run_pending()
        worker = Worker(func, self)
        if self.spectra is not None:
            self._apply_temp_dir()
            self.spectra._progress_callback = worker.report
        worker.progress.connect(self._on_progress)
        worker.done.connect(lambda result: self._finish(on_done, result))
        worker.failed.connect(self._on_failed)
        self.worker = worker
        self.progress.setRange(0, 0)  # busy until the first progress report
        self.progress.setVisible(True)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.statusBar().showMessage(message)
        self.refresh()
        worker.start()

    def _on_progress(self, n: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(n)

    def _end_worker(self) -> None:
        if self.spectra is not None:
            self.spectra._progress_callback = None
        if self.worker is not None:
            self.worker.wait()
        self.worker = None
        self.progress.setVisible(False)
        self.cancel_button.setVisible(False)

    def _finish(self, on_done: Callable[[Any], None], result: Any) -> None:
        self._end_worker()
        self.statusBar().showMessage('done', 5_000)
        try:
            on_done(result)
        finally:
            self.refresh()

    def _on_failed(self, message: str) -> None:
        self._end_worker()
        if message == 'cancelled':
            self.statusBar().showMessage('cancelled, results of the step are incomplete', 15_000)
        else:
            logger.error(message)
            QtWidgets.QMessageBox.critical(self, 'Error', message)
        self.refresh()

    def _cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancelled = True
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage('cancelling ...')

    def run_sum(self) -> None:
        s: Spectra = self.spectra
        recalibrate: bool = self.sum_calibrated.isChecked()
        lost: list[str] = self._results_after('sum')
        if not self._confirm_reset(lost, 'Summing the spectra again'):
            return
        n_jobs: int = self.n_jobs.value()

        def job() -> np.ndarray[float]:
            s.add_all_spectra(self.reader, recalibrate=recalibrate, n_jobs=n_jobs)
            return s._intensities.copy()

        def done(raw: np.ndarray[float]) -> None:
            first: bool = self.raw_intensities is None
            self.raw_intensities = raw
            self.sum_is_calibrated = recalibrate
            if not recalibrate:
                self.uncalibrated_sum = raw
            self.applied['sum'] = {}
            self.reset_notes.pop('sum', None)
            self._note_reset(lost, 'spectra summed again')
            self.peaks_state = self.binned_state = None
            self.tabs.setCurrentWidget(self.plot)
            if first:
                self.refresh()
                self.plot.reset_view()

        self._start(job, done, 'summing spectra')

    def run_calibration(self) -> None:
        s: Spectra = self.spectra
        kwargs: dict[str, Any] = self._calibration_kwargs()
        if (kwargs['calib_snr_threshold'] > 0) and (not self._has('noise')):
            QtWidgets.QMessageBox.warning(self, 'Calibration', 'estimate the noise level first or set the SNR '
                                                               'threshold to 0')
            return
        lost: list[str] = self._results_after('calibration')
        if not self._confirm_reset(lost, 'Calibrating (replaces the summed spectrum)'):
            return
        if not self.sum_is_calibrated:
            self.uncalibrated_sum = self.raw_intensities
        n_jobs: int = self.n_jobs.value()

        def job() -> np.ndarray[float]:
            s.set_calibration_functions(self.reader, n_jobs=n_jobs, add_spectra=True, **kwargs)
            return s._intensities.copy()

        def done(raw: np.ndarray[float]) -> None:
            self.raw_intensities = raw
            self.sum_is_calibrated = True
            self.sum_calibrated.setChecked(True)
            self.applied['calibration'] = kwargs
            self.reset_notes.pop('calibration', None)
            # the noise level has to be estimated again from the calibrated sum
            self._note_reset(lost, 'spectra calibrated')
            self.peaks_state = self.binned_state = None
            self.refresh()
            self.show_calibration_summary()

        self._start(job, done, 'finding calibration functions and summing calibrated spectra')

    def run_kernels(self) -> None:
        s: Spectra = self.spectra
        if self.peaks_state is None:
            return
        lost: list[str] = self._results_after('kernels')
        if not self._confirm_reset(lost, 'Fitting the kernels again'):
            return
        params: dict[str, Any] = self._params('kernels')
        peaks_state: dict[str, Any] = copy.deepcopy(self.peaks_state)

        def job() -> tuple[int, int, int]:
            # start from the peaks of the peak step, peaks removed by a previous fit come back
            for name, value in peaks_state.items():
                setattr(s, name, value)
            s.reset_kernels()
            n_peaks: int = len(s._peaks)
            s.set_kernels(use_bigaussian=params['bigaussian'], fine_tune=params['fine_tune'],
                          discard_invalid=params['discard_invalid'], sigma_max=params['sigma_max'] * 1e-3)
            n_fit: int = len(s._peaks)
            if params['thr_sigma'] is not None:
                s.filter_peaks(thr_sigma=params['thr_sigma'])
            return n_peaks, n_fit, len(s._peaks)

        def done(counts: tuple[int, int, int]) -> None:
            self.kernel_counts = counts
            self.applied['kernels'] = params
            self.reset_notes.pop('kernels', None)
            self._note_reset(lost, 'kernels fitted again')
            self.binned_state = None

        self._start(job, done, 'fitting kernels')

    def run_binning(self) -> None:
        s: Spectra = self.spectra
        method: str = self.bin_method.currentText()
        if (method == 'max') and (s._kernel_shape != 'gaussian'):
            QtWidgets.QMessageBox.warning(self, 'Binning', 'max is only implemented for gaussian kernels')
            return
        n_jobs: int = self.n_jobs.value()

        def job() -> None:
            s.bin_spectra(self.reader, method=method, n_jobs=n_jobs, show_progress=False)

        def done(_: Any) -> None:
            self._store_binned_state()
            self.applied['binning'] = dict(method=method)
            self.reset_notes.pop('binning', None)
            # filter with the current parameters
            self._apply_filter()
            self.applied['filter'] = self._params('filter')
            self.tabs.setCurrentWidget(self.binned_tab)

        self._start(job, done, 'binning spectra')

    # ------------------------------------------------------------------ results
    def save(self) -> None:
        s: Spectra = self.spectra
        s.set_feature_table()
        try:
            s.save()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, 'Save', f'could not save: {e}')
            return
        self.statusBar().showMessage(f'saved feature table with {s.feature_table.shape} entries', 15_000)

    def parameters_as_code(self) -> str:
        """Code for processing spectra s with reader with the selected parameters."""
        calibration: dict[str, Any] = self._calibration_kwargs()
        window: str = 'None' if self.noise_auto.isChecked() else f'{self.noise_window.value():g}'
        lines: list[str] = [
            f'n_jobs = {self.n_jobs.value()}',
        ]
        if self.temp_edit.text().strip():
            lines.append(f's.set_temp_dir(r{self.temp_edit.text().strip()!r})')
        lines += [
            's.add_all_spectra(reader=reader, recalibrate=False, n_jobs=n_jobs)',
            f's.set_noise_level(window_size={window})',
            's.set_calibration_functions(',
            '    reader=reader,',
            f'    calibrants_mz={[round(mz, 6) for mz in calibration["calibrants_mz"]]},',
            f'    search_range={calibration["search_range"]:g},',
            f'    calib_snr_threshold={calibration["calib_snr_threshold"]:g},',
            f'    min_height={calibration["min_height"]:g},',
            f'    max_degree={calibration["max_degree"]},',
            f'    nearest={calibration["nearest"]},',
            '    n_jobs=n_jobs,',
            '    add_spectra=True',
            ')',
            f's.set_noise_level(window_size={window})',
            's.subtract_baseline()',
            f's.set_peaks(prominence={self.prominence.value():g}, width={self.peak_width.value()}'
            + (f', distance={self.peak_distance.value()})' if self.peak_distance.value() else ')'),
        ]
        if (self.peaks_snr.value() > 0) or self.remove_side_peaks.isChecked():
            lines.append(f's.filter_peaks(peaks_snr_threshold={self.peaks_snr.value():g}, '
                         f'remove_sidepeaks={self.remove_side_peaks.isChecked()})')
        lines.append(f's.set_kernels(use_bigaussian={self.bigaussian.isChecked()}, '
                     f'fine_tune={self.fine_tune.isChecked()}, discard_invalid={self.discard_invalid.isChecked()}, '
                     f'sigma_max={self.fit_sigma_max.value() * 1e-3:g})')
        if (thr_sigma := self._thr_sigma()) is not None:
            lines.append(f's.filter_peaks(thr_sigma=({thr_sigma[0]:g}, {thr_sigma[1]:g}))')
        lines += [
            f's.bin_spectra(reader=reader, method={self.bin_method.currentText()!r}, n_jobs=n_jobs)',
            f's.filter_line_spectra(binned_snr_threshold={self.binned_snr.value():g}, '
            f'intensity_min={self.intensity_min.value():g}, success_rate_threshold={self.success_rate.value():g})',
            's.set_feature_table()',
        ]
        if self.temp_edit.text().strip():
            lines.append('s.remove_temp_dir()')
        return '\n'.join(lines)

    def show_code(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle('Parameters as code')
        dialog.resize(800, 500)
        text = QtWidgets.QPlainTextEdit(self.parameters_as_code())
        text.setReadOnly(True)
        text.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))
        copy_button = QtWidgets.QPushButton('Copy to clipboard')
        copy_button.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(text.toPlainText()))
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(text)
        layout.addWidget(copy_button)
        dialog.exec_()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.worker is not None:
            answer = QtWidgets.QMessageBox.question(self, 'Close', 'A step is still running, cancel it and close?')
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self.worker.cancelled = True
            self.worker.wait()
        if self.spectra is not None:
            self.spectra._progress_callback = None
        if (self.spectra is not None) and (self._temp_dir_set is not None):
            self.spectra.remove_temp_dir()
            self._temp_dir_set = None
        logging.getLogger('maspim').removeHandler(self._log_handler)
        if sys.excepthook == self._excepthook:
            sys.excepthook = sys.__excepthook__
        super().closeEvent(event)


def _qt_event_loop_running() -> bool:
    """Whether IPython runs a Qt event loop (e.g. %gui qt), so that the window can be shown without blocking."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return (shell is not None) and (getattr(shell, 'active_eventloop', None) in ('qt', 'qt5'))


def run_spectra_gui(
        spectra: Spectra | None = None,
        reader: Any = None,
        n_jobs: int = 8,
        path_temp: str | None = None,
        path_folder: str | None = None
) -> SpectraGui:
    """
    Open the GUI (see SpectraGui), optionally for spectra that are already loaded.

    Blocks until the window is closed, unless a Qt event loop is running already
    (e.g. in IPython with %gui qt). The results of the selected parameters are
    stored in the spectra (gui.spectra for a measurement opened in the window).
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    gui = SpectraGui(spectra, reader, n_jobs=n_jobs, path_temp=path_temp, path_folder=path_folder)
    gui.show()
    if not _qt_event_loop_running():
        app.exec_()
    return gui


def main() -> None:
    """Open the GUI, the measurement and all parameters are selected in the window."""
    logging.basicConfig(level=logging.INFO)
    run_spectra_gui()


if __name__ == '__main__':
    main()
