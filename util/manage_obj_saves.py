"""
This module defines which attributes are saved for each object when the save method is called.

Using this strategy ensures that 'clean' instances of objects are saved but will remove custom user attributes.
Also, this removes redundancy for the Image related classes: it is useful to set the original image as an attribute
but not to save it on disc because the image file itself is already stored on disc somewhere. Hence, only saving the
path to the image file on disc is sufficient and equally fast to load.

"""
import logging

logger = logging.getLogger(__name__)

Image_disc_attributes = {
    'path_image_file',
    'age_span',
    'image_file'
}

ImageSample_disc_attributes = Image_disc_attributes | {
    'obj_color', '_xywh_ROI', '_hw'}

ImageROI_disc_attributes = ImageSample_disc_attributes | {
    '_image_classification', '_params', '_punchholes', '_punchhole_size', '_image_original'
}

ImageClassified_disc_attributes = ImageSample_disc_attributes | {
    'params_laminae_simplified',
    'image_seeds',
    '_image_classification',
    '_image_original'
}

ImageTransformation_disc_attributes = {}

Data_disc_attributes = {
    'distance_pixels',
    'feature_table',
    'depth_section',
    'age_span'
}

Data_nondata_columns = {
    'L',
    'x_ROI',
    'y_ROI',
    'valid',
    'classification',
    'classification_s',
    'classification_se',
    'seed',
    'depth',
    'age',
    'x',
    'y'
}

MSI_disc_attributes = Data_disc_attributes | {'d_folder', 'mis_file'}
XRF_disc_attributes = Data_disc_attributes | {
    'default_file_type', 'measurement_name', 'prefix_files'
}

TimeSeries_disc_attributes = {
    'feature_table',
    'feature_table_standard_deviations',
    'feature_table_successes'
}

MetaFeatures_disc_attributes = {
    'cmpds',
    'formulas',
    'df',
    'features_clustering'
}

SampleImageHandlerMSI_attributes = {'extent_spots', 'd_folder', 'mis_file', 'image_file'}

SampleImageHandlerXRF_attributes = {
    'image_file',
    'image_roi_file',
    'ROI_is_image',
    'extent_spots',
    'extent',
    'scale_conversion'
}

Spectra_attributes = {
    'd_folder',
    'delta_mz',
    'mzs',
    'intensities',
    'indices',
    'limits',
    'peaks',
    'peak_properties',
    'peak_setting_parameters',
    'kernel_shape',
    'kernel_params',
    'line_spectra',
    'feature_table',
    'losses',
    'binning_by',
    'noise_level',
    'calibration_parameters',
    'calibration_settings'
}
MultiSectionSpectra_attributes = Spectra_attributes

XRay_attributes = {
    'image_file',
    'depth_section',
    'obj_color',
    'depth_section',
    '_image',
    '_xywh_ROI',
    '_image_ROI',
    '_bars_removed'
}

Transformation_attributes = {
    'source', 'target', 'target_shape', 'source_shape', 'trafos', 'trafo_types'
}

Mapper_attributes = {'_Us', '_Vs', '_image_shape', '_tag'}

name_to_attrs = {
    'Image': Image_disc_attributes,
    'ImageSample': ImageSample_disc_attributes,
    'ImageROI': ImageROI_disc_attributes,
    'ImageClassified': ImageClassified_disc_attributes,
    'Data': Data_disc_attributes,
    'MSI': MSI_disc_attributes,
    'XRF': XRF_disc_attributes,
    'TimeSeries': TimeSeries_disc_attributes,
    'MetaFeatures': MetaFeatures_disc_attributes,
    'Spectra': Spectra_attributes,
    'XRay': XRay_attributes,
    'SampleImageHandlerMSI': SampleImageHandlerMSI_attributes,
    'SampleImageHandlerXRF': SampleImageHandlerXRF_attributes,
    'MultiSectionSpectra': MultiSectionSpectra_attributes,
    'Transformation': Transformation_attributes,
    'Mapper': Mapper_attributes
}


def class_to_attributes(class_object: object) -> set[str]:
    c: str = class_object.__class__.__name__
    attrs = name_to_attrs.get(c)
    if attrs is None:
        logger.info(f'found no entry for {c}, keeping all attributes')
        attrs = class_object.__dict__.keys()
    return attrs
