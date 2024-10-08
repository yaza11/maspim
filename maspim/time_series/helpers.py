import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from maspim.data.msi import MSI
from maspim.data.xrf import XRF
from maspim.imaging.main import ImageClassified
from maspim.util.convenience import check_attr

logger = logging.getLogger(__name__)


def get_averaged_tables(
        data_object: MSI | XRF,
        image_classified: ImageClassified,
        is_continuous: bool,
        average_by_col: str = 'classification_se',
        exclude_holes: bool = True,
        holes_column: str | float | int = 'valid',
        holes_key: str | float | int = 0,
        plts: bool = False,
        **kwargs
) -> tuple[pd.DataFrame, ...]:
    assert isinstance(data_object, MSI | XRF)
    assert check_attr(data_object, '_feature_table')
    feature_table = data_object.feature_table

    if (
            (not is_continuous) and (
            (image_classified is None) or
            not check_attr(image_classified, 'params_laminae_simplified'))
    ):
        raise ValueError(
            'ImageClassified does not have laminae parameters set. Cannot '
            'filter laminae based on that'
        )

    assert average_by_col in feature_table.columns, \
        ('call add_laminae_classification or use a different column name to'
         ' average by')

    if exclude_holes:
        if holes_column not in data_object.columns:
            logging.warning(
                f'cannot use {holes_column=} because the column is not in the '
                f'feature table, proceeding with not excluding holes'
            )
            exclude_holes = False
    if exclude_holes:
        # TODO: test
        # set rows with holes to nan in copy
        data_object: MSI | XRF = data_object.copy()
        mask_holes: pd.Series = (
                data_object.feature_table.loc[:, holes_column] == holes_key
        )
        # all columns except the average by column
        columns: pd.Series = (
            data_object.feature_table.drop(columns=average_by_col).columns
        )
        # set to nan to exclude from average
        data_object.feature_table.loc[mask_holes, columns] = np.nan
        logging.info(
            f'excluding {mask_holes.sum()} data points that were identified as '
            'holes'
        )
    else:
        columns = data_object.feature_table.columns

    ft_seeds_avg, ft_seeds_std, ft_seeds_success = data_object.processing_zone_wise_average(
        zones_key=average_by_col,
        columns=columns,
        calc_std=kwargs.pop('calc_std', True),
        exclude_zeros=kwargs.pop('exclude_zeros', True),
        **kwargs
    )

    ft_seeds_avg = ft_seeds_avg.fillna(0)

    if not is_continuous:
        # add quality criteria
        cols_quals = ['homogeneity', 'continuity', 'contrast', 'quality']
        # only consider those seeds that are actually in the image
        seeds = image_classified.params_laminae_simplified.seed.copy()
        seeds *= np.array([
            1 if (c == 'light') else -1
            for c in image_classified.params_laminae_simplified.color]
        )
        row_mask = [seed in ft_seeds_avg.index for seed in seeds]

        quals = image_classified.params_laminae_simplified.loc[
            row_mask, cols_quals + ['height']
        ].copy()
        # take weighted average for laminae with same seeds (weights are areas=heights)
        quals_weighted = quals.copy().mul(quals.height.copy(), axis=0)
        # reset height (otherwise height column would have values height ** 2
        quals_weighted['height'] = quals.height.copy()
        quals_weighted['seed'] = seeds
        quals_weighted = quals_weighted.groupby(by='seed').sum()
        quals_weighted = quals_weighted.div(quals_weighted.height, axis=0)
        quals_weighted.drop(columns=['height'], inplace=True)

        # join the qualities to the averages table
        ft_seeds_avg = ft_seeds_avg.join(quals_weighted, how='left')
        # insert infty for every column in success table that is not there yet
        missing_cols = set(ft_seeds_avg.columns).difference(
            set(ft_seeds_success.columns)
        )
        for col in missing_cols:
            ft_seeds_success.loc[:, col] = np.inf

        # plot the qualities
        if plts:
            plt.figure()
            plt.plot(quals_weighted.index,
                     quals_weighted.quality,
                     '+',
                     label='qual')
            plt.plot(ft_seeds_avg.index, ft_seeds_avg.quality, 'x', label='ft')
            plt.legend()
            plt.xlabel('zone')
            plt.ylabel('quality')
            plt.title('every x should have a +')
            plt.show()

    # drop index (=seed) into dataframe
    ft_seeds_avg.index.names = ['zone']
    ft_seeds_std.index.names = ['zone']
    ft_seeds_success.index.names = ['zone']
    # reset index
    ft_seeds_avg.reset_index(inplace=True)
    ft_seeds_std.reset_index(inplace=True)
    ft_seeds_success.reset_index(inplace=True)

    # change column type
    if not is_continuous:
        ft_seeds_avg.loc[:, 'zone'] = ft_seeds_avg.zone.astype(int)
        ft_seeds_std.loc[:, 'zone'] = ft_seeds_std.zone.astype(int)
        ft_seeds_success.loc[:, 'zone'] = ft_seeds_success.zone.astype(int)

    # need to insert the x_ROI from avg
    ft_seeds_std['spread_x_ROI'] = ft_seeds_std.x_ROI.copy()
    ft_seeds_std['x_ROI'] = ft_seeds_avg.x_ROI.copy()

    ft_seeds_success['N_total'] = ft_seeds_success.x_ROI.copy()
    ft_seeds_success['x_ROI'] = ft_seeds_avg.x_ROI.copy()

    # sort by depth
    ft_seeds_avg = ft_seeds_avg.sort_values(by='x_ROI')
    ft_seeds_std = ft_seeds_std.sort_values(by='x_ROI')
    ft_seeds_success = ft_seeds_success.sort_values(by='x_ROI')

    # drop columns with seed == 0
    if 'classification' in average_by_col:
        mask_drop = ft_seeds_avg.zone == 0
        ft_seeds_avg.drop(index=ft_seeds_avg.index[mask_drop], inplace=True)
        ft_seeds_std.drop(index=ft_seeds_std.index[mask_drop], inplace=True)
        ft_seeds_success.drop(index=ft_seeds_success.index[mask_drop], inplace=True)

    # reset index
    ft_seeds_avg.reset_index(inplace=True, drop=True)
    ft_seeds_std.reset_index(inplace=True, drop=True)
    ft_seeds_success.reset_index(inplace=True, drop=True)

    return ft_seeds_avg, ft_seeds_success, ft_seeds_std
