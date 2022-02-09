use gdal::errors::GdalError;
use geoengine_datatypes::dataset::DatasetProviderId;
use snafu::Snafu;

// TODO: `Clone` and `PartialEq`

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
#[snafu(context(suffix(false)))] // disables default `Snafu` suffix
pub enum NetCdfCf4DProviderError {
    NotYetImplemented,
    DataTypeNotYetImplemented {
        data_type: String,
    },
    MissingTimeCoverageStart {
        source: GdalError,
    },
    MissingTimeCoverageEnd {
        source: GdalError,
    },
    MissingTimeCoverageResolution {
        source: GdalError,
    },
    MissingTitle {
        source: GdalError,
    },
    MissingCrs {
        source: GdalError,
    },
    MissingSubdatasets,
    MissingEntities {
        source: GdalError,
    },
    MissingGroupName,
    MissingFileName,
    NoTitleForGroup {
        metadata_key: String,
    },
    CannotParseNumberOfEntities {
        source: std::num::ParseIntError,
    },
    CannotSplitNumberOfEntities,
    CannotConvertTimeCoverageToInt {
        source: std::num::ParseIntError,
    },
    GeneratingResultDescriptorFromDataset {
        source: geoengine_operators::error::Error,
    },
    GeneratingParametersFromDataset {
        source: geoengine_operators::error::Error,
    },
    InvalidTimeCoverageInterval {
        source: geoengine_datatypes::error::Error,
    },
    CannotCalculateStepsInTimeCoverageInterval {
        source: geoengine_datatypes::error::Error,
    },
    InvalidExternalDatasetId {
        provider: DatasetProviderId,
    },
    InvalidDatasetIdLength {
        length: usize,
    },
    InvalidDatasetIdFile {
        source: geoengine_operators::error::Error,
    },
    CannotParseCrs {
        source: geoengine_datatypes::error::Error,
    },
    DatasetIdEntityNotANumber {
        source: std::num::ParseIntError,
    },
    CannotComputeSubdatasetsFromMetadata,
    GdalMd {
        source: GdalError,
    },
    UnknownGdalDatatype {
        type_number: u32,
    },
    MustBe4DDataset {
        number_of_dimensions: usize,
    },
    CannotGetGeoTransform {
        source: GdalError,
    },
    InvalidGeoTransformLength {
        length: usize,
    },
    InvalidGeoTransformNumbers {
        source: std::num::ParseFloatError,
    },
}
