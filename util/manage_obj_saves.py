
Image_disc_attributes = {
    'path_folder',
    'age_span'
    }

ImageSample_disc_attributes = Image_disc_attributes | {
    'obj_color', 'xywh_ROI'}

ImageROI_disc_attributes = ImageSample_disc_attributes | {
    'image_classification', 'params_classification'
}

ImageClassified_disc_attributes = ImageSample_disc_attributes | {
    'params_laminae_simplified',
    'image_seeds'}

ImageTransformation_disc_attributes = {'_section, _window_from', '_window_to'}

Data_disc_attributes = {
    'path_d_folder',
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
    'age'
}

MSI_disc_attributes = Data_disc_attributes
XRF_disc_attributes = Data_disc_attributes | {'default_file_type'}

TimeSeries_disc_attributes = {
    'path_folder',
    'feature_table',
    'feature_table_standard_deviations',
    'feature_table_successes'
}

MetaFeatures_disc_attributes = {
    '_section',
    'windows',
    '_window',
    'cmpds',
    'formulas',
    'df',
    'features_clustering'
}

SampleImageHandler_attributes = {'extent_spots', 'path_folder'}

Spectra_attributes = {
    'path_d_folder', 
    'delta_mz', 'mzs', 
    'intensities', 
    'indices', 
    'limits', 
    'peaks',
    'peak_properties',
    'peak_setting_parameters',
    'kernel_shape',
    'kernel_params',
    'line_spectra'
}

XRay_attributes = {
    'path_image_file',
    'depth_section',
    'obj_color',
    'depth_section',
    'xywh_ROI',
    'image_ROI'
}

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
    'SampleImageHandler': SampleImageHandler_attributes,
    'Spectra': Spectra_attributes,
    'XRay': XRay_attributes
}


def class_to_attributes(class_object: object) -> list[str]:
    c: str = class_object.__class__.__name__
    return name_to_attrs[c]
