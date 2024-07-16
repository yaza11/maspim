raise NotImplementedError()

from src.res.constants import elements, max_deviation_mz
from src.res.directory_paths import file_dataBase
# python
from typing import Iterable
from dataclasses import dataclass
import re
# standard
import numpy as np
import pandas as pd
import scipy.constants
# third party
import pyteomics
from pyteomics import mass as mpyteomics
from pykrev import kendrick_mass_defect


m_e = scipy.constants.value('electron mass in u')

explainable_mass_interval_types = [
    'H', 'C[13]', 'H2', 'O', 'H2O', 'OH', 'CO', 'N2', 'NO2',
    'CH2', 'CH3', 'C2H2', 'C2H3'
]
adduct_types = ['M+', 'H+', 'Na+', 'K+', 'NH4+']


def formula_to_mass(formula: str, protected=False):
    # get list of elements and their abundance
    ionization = re.findall(r'[+, -]\d*', formula)
    if len(ionization) > 1:
        if protected:
            return -1
        else:
            raise ValueError(f'found invalid ionization: {ionization}')
    elif len(ionization) == 1:
        ionization = ionization[0]
        if len(ionization) > 1:
            count = int(ionization[1:])
        else:
            count = 1

        if '+' in ionization:
            dm = -m_e * count
            formula = formula[:-len(ionization)]
        else:
            dm = +m_e * count
            formula = formula[:-len(ionization)]
    else:
        dm = 0
    try:
        m = mpyteomics.calculate_mass(formula=formula)
        return m + dm
    # except Exception as e:
    #     if protected:
    #         return -1
    #     else:
    #         raise e
    except:
        pass


explainable_mass_intervals = np.array([
    formula_to_mass(group)
    if group != 'C[13]'
    else formula_to_mass(group) - formula_to_mass('C')
    for group in explainable_mass_interval_types
], dtype=float)
adduct_masses = np.array([
    formula_to_mass(group)
    if group != 'M+'
    else formula_to_mass('+')
    for group in adduct_types
], dtype=float)


class DataBase:
    def __init__(self, path):
        self.path = path
        self._DB = None
        self._DB_all = None

    def _load(self):
        DB = pd.read_csv(file_dataBase, index_col=0)
        self._DB = DB.groupby('Input m/z').first().reset_index()

    def _load_all(self):
        DB = pd.read_csv(file_dataBase, index_col=0)
        self._DB_all = DB

    def _get(self):
        if self._DB is None:
            self._load()
        return self._DB

    def _get_all(self):
        if self._DB_all is None:
            self._load_all()
        return self._DB_all

    def find(self, mass):
        try:
            idx = self._get().loc[:, 'Input m/z'] == float(mass)
            return self._get().loc[idx, 'Formula'].iat[0]
        except:
            return str(mass)

    def find_all(self, mass):
        rows = self._get_all().loc[:, 'Input m/z'] == float(mass)
        return self._get_all().loc[rows, :].sort_values(by='Delta')\
            .reset_index(drop=True)


dataBase = DataBase(file_dataBase)


@dataclass
class Annotation:
    mz: float
    precision: float
    derivates: np.ndarray[float]
    modifiers: np.ndarray[str]
    db_entry: pd.DataFrame
    adducts: np.ndarray[str]

    def has_child(self):
        return len(self.derivates) > 0

    def get_corrected_formula(self, f, ion):
        if abs(formula_to_mass(f + '+', protected=True) - self.mz) <= self.precision:
            return f + '+'
        elif abs(formula_to_mass(f + ion + '+', protected=True) - self.mz) <= self.precision:
            return f + ion + '+'
        return None

    def name(self):
        """Can be suffisticated later, for now first entry in db."""
        fs = self.db_entry['Formula']
        ions = self.db_entry['Ion']
        for i, f in enumerate(fs):
            match = re.findall(r'\[M\+(.+)\]\+', ions.iat[i])
            if len(match) == 1:
                ion = match[0]
            else:
                ion = ''
            if (f_corrected := self.get_corrected_formula(f, ion)) is not None:
                return f_corrected
        return str(round(self.mz, 4))


class Annotations:
    def __init__(
            self,
            mzs: Iterable[float | str],
            mz_precision: float,
            intensities: Iterable | None = None,
            ionization_type='positive'
    ):
        """Initiate class attributes."""
        self.mzs = np.array(mzs).astype(float)
        self.mzs.sort()

        assert len(self.mzs) == len(np.unique(self.mzs)), 'masses contain duplicate entries'

        self.precision = mz_precision
        self.ionization_type = ionization_type
        if (intensities is not None) and (mzs is not None):
            assert (len(intensities) == len(mzs)) or (intensities.shape[-1] == len(mzs)), 'mzs and intensities must have same length'
        if intensities is not None:
            self.intensites = np.array(intensities).astype(float)
        else:
            self.intensites = None

        # # search for known intervals
        self.find_known_intervals()
        del self._lookup

    def lookup(self):
        if not hasattr(self, '_lookup'):
            L = np.outer(
                self.mzs,
                np.ones(len(explainable_mass_intervals))
            )
            L += np.array(explainable_mass_intervals)
            self._lookup = L
        return self._lookup

    def find_known_intervals(self):
        """Search the list of mzs for known intervals."""
        # dict_mz = dict(zip(list_mz, list_mz))
        self.derivates = []  # container for possible derivatives, parent mzs are always lighter, children have added functional groups
        self.modifiers = []  # container for the type of derivative
        for i, mz in enumerate(self.mzs):
            # check if compound has explainable mass difference to other known compound
            deviations = np.abs(self.lookup() - mz)
            mask_valid = deviations <= self.precision * 2
            if not np.any(mask_valid):
                self.derivates.append(np.array([]))
                self.modifiers.append(np.array([]))
                continue
            # get tuple indices
            idxs = np.argwhere(mask_valid)
            # list of possible mz_origs
            mz_orig_list = [
                self.lookup()[idx[0], idx[1]] - explainable_mass_intervals[idx[1]]
                for idx in idxs
            ]
            dists = [np.abs(self.lookup()[idx[0], idx[1]] - mz) for idx in idxs]
            o = np.argsort(dists)
            # list of corresponding modifications
            add_list = [explainable_mass_interval_types[idx[1]] for idx in idxs]
            # remove C13 if derived compound is lighter than orig compound
            for mz_orig, add in zip(mz_orig_list, add_list):
                if (add == 'C[13]') and (mz_orig > mz):
                    mz_orig_list.remove(mz_orig)
                    add_list.remove(add)
            self.derivates.append(np.array(mz_orig_list)[o])
            self.modifiers.append(np.array(add_list)[o])

    def search_other_adducts(self, mz):
        # TODO: add search for other adducts
        pass

    def querry_data_base(self, mz):
        return dataBase.find_all(mz)

    def get_for(self, mz):
        idx = np.argwhere(self.mzs == float(mz))[0][0]
        derivates = self.derivates[idx]
        modifiers = self.modifiers[idx]
        db_entry = self.querry_data_base(mz)
        adducts = self.search_other_adducts(mz)
        annotation = Annotation(
            mz=float(mz),
            precision=self.precision,
            derivates=derivates,
            modifiers=modifiers,
            db_entry=db_entry,
            adducts=adducts
        )
        return annotation


def get_matching_formulas(db_a, db_b, modification):
    """Search for possible formulas such that a + modification = b."""
    # unique formulas in a
    form_a = list(np.unique(db_a.Formula))
    # lookup
    form_b = set(db_b.Formula.tolist())
    # pyteomics composition object of modification
    mod = mpyteomics.mass.Composition(formula=modification)
    # container for pyteomics composition object
    pyf_b = []
    for fb in form_b:
        if not (type(fb) is str):
            continue
        try:
            pyf_b.append(mpyteomics.mass.Composition(formula=fb))
        except Exception as e:
            print(e)
    # look if a  + mod in b
    possible_combos = []
    for idx, fa in enumerate(form_a):
        if not (type(fa) is str):
            continue
        try:
            pyf_a = mpyteomics.mass.Composition(formula=fa)
            if pyf_a + mod in pyf_b:
                possible_combos.append(form_a[idx])
        except Exception as e:
            print(e)
    return possible_combos


class AllAnotations(Annotations):
    def __init__(
            self,
            mz_precision: float,
            ionization_type='positive'
    ):
        self.precision = mz_precision
        self.ionization_type = ionization_type

        self.initiate_mzs()
        self.find_known_intervals()
        del self._lookup

    def initiate_mzs(self):
        from data.cMSI import MSI
        mzs = []
        section = (490, 495)
        for window in ('FA', 'Alkenones', 'GDGT'):
            m = MSI(section, window)
            if window in ('FA', 'Alkenones'):
                m.load()
            else:
                m.load(use_common_mzs=True)
            m.current_feature_table
            mzs += list(m.get_data_columns().copy())
            del m
        self.mzs = np.array(mzs).astype(float)


def find_closests(values: np.ndarray, targets: np.ndarray):
    assert np.all(~np.isnan(values)), 'values contain nan'
    assert np.all(~np.isnan(targets)), 'targets contain nan'
    # Broadcasting to create a 2D array of absolute differences
    abs_diffs = np.abs(values[:, np.newaxis] - targets)

    # Finding the index of the closest value for each target
    closest_indices = np.argmin(abs_diffs, axis=0)

    # Retrieving the closest values for each target
    closest_values = values[closest_indices]
    return closest_values


class Homologous:
    def __init__(
            self,
            mzs: Iterable,
            intensities: Iterable | None = None,
            repeating_units: Iterable = ['CH2'],  # repeating units making up series
            n_series_min: int = 3  # min members in series to be considered
    ):
        self.mzs = np.array(mzs).astype(float)
        if intensities is not None:
            self.intensities = np.array(intensities).astype(float)
        self.repeating_units = list(repeating_units)
        self.n_series_min = n_series_min

    def find_series(self, include_missings: bool = True):
        """Identify series of homologs in mz list.

        include_missings: bool
            if this is true, missing mzs will be denoted by nan, otherwise series
            only includes found mzs
        """
        self.series = {}
        for repeating_unit in self.repeating_units:
            print(f'looking for homologs with {repeating_unit}')
            mzs_left = list(self.mzs.copy())
            dmz = mpyteomics.calculate_mass(formula=repeating_unit)
            for mz_seed in self.mzs:
                print(f'taking seed {mz_seed}, looking in {len(mzs_left)} compounds')
                s = []
                if (mz_seed not in mzs_left) or (len(mzs_left) < self.n_series_min):
                    continue

                mzs_left.remove(mz_seed)
                # try to find homologs
                i_max = np.abs(mz_seed - self.mzs.max()) // dmz + 1
                mz_targets = np.arange(mz_seed, mz_seed + (i_max + 1) * dmz, dmz)
                mz_closests = find_closests(self.mzs, mz_targets)
                hits = np.abs(mz_closests - mz_targets) < 2 * max_deviation_mz
                if np.sum(hits) < self.n_series_min:
                    continue

                if include_missings:
                    s = mz_closests
                    s[~hits] = np.nan
                else:
                    s = mz_closests[hits]
                mzs_left = list(set(mzs_left).difference(set(mz_closests[hits])))
                self.series[(repeating_unit, mz_seed)] = s

    def find_formula(self):
        from pyteomics.mass import Composition as comp
        db = Annotations(self.mzs.copy(), max_deviation_mz)

        def comp_to_f(c):
            f = ''.join([''.join([i[0], str(i[1])]) for i in c.items()])
            return f

        series = {}
        for k, v in self.series.items():
            repeating_unit = k[0]
            # try to find mass with database entry within series
            formulas = set()
            found_entry: bool = False
            for idx, mz in enumerate(v):
                e = db.querry_data_base(mz)
                if len(e) > 0:
                    fs = set([comp_to_f(comp(f) - comp(repeating_unit) * idx) for f in e.Formula])
                    if not found_entry:
                        formulas = fs
                        found_entry = True
                    else:
                        formulas &= fs
            if len(formulas) > 0:
                k = (repeating_unit, list(formulas)[0])
                print(formulas)
            series[k] = v
        self.series = series


def test_homolog_series():
    dmz = mpyteomics.calculate_mass(formula='CH2')

    mzs1 = np.arange(10, 200, dmz)
    mzs2 = np.arange(12, 200, dmz)
    mzs = np.concatenate([mzs1, mzs2])
    mzs += 2 * (np.random.random(mzs.shape) - .5) * 2 / 1000
    hs = Homologous(mzs=mzs)
    hs.find_series()
    print(hs.series)
    return hs


def test_homolog_series_FA_upper():
    import pickle
    with open('E:/Master_Thesis/raw_feature_tables/490-520/FA/ref_peaks_after.pickle', 'rb') as f:
        mzs = pickle.load(f)
    hs = Homologous(mzs)
    hs.find_series(include_missings=False)
    # make sure mz is not in two series
    ms = []
    for a in hs.series.values():
        ms_ = [m for m in a if str(m).replace('.', '').isnumeric()]
        ms.extend(ms_)
    print(len(ms))
    print(len(np.unique(ms)))


if __name__ == '__main__':
    # a = AllAnotations(max_deviation_mz)
    pass
