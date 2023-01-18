use geoengine_datatypes::{
    collections::DataCollection,
    primitives::{FeatureData, Measurement, NoGeometry, TimeInterval},
    raster::{Grid2D, MaskedGrid2D, RasterDataType, RasterTile2D, TileInformation},
    spatial_reference::SpatialReference,
    util::test::TestDefault,
};

use crate::{
    engine::{RasterResultDescriptor, SourceOperator, VectorOperator},
    mock::{MockFeatureCollectionSource, MockRasterSource, MockRasterSourceParams},
    util::Result,
};

pub fn generate_vector_test_data_band_helper() -> Result<Box<dyn VectorOperator>> {
    Ok(MockFeatureCollectionSource::multiple(vec![
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![1])),
                ("precipitation", FeatureData::Int(vec![9])),
                ("target", FeatureData::Int(vec![0])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![2])),
                ("precipitation", FeatureData::Int(vec![10])),
                ("target", FeatureData::Int(vec![1])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![3])),
                ("precipitation", FeatureData::Int(vec![11])),
                ("target", FeatureData::Int(vec![2])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![4])),
                ("precipitation", FeatureData::Int(vec![12])),
                ("target", FeatureData::Int(vec![2])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![5])),
                ("precipitation", FeatureData::Int(vec![13])),
                ("target", FeatureData::Int(vec![2])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![6])),
                ("precipitation", FeatureData::Int(vec![14])),
                ("target", FeatureData::Int(vec![1])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![7])),
                ("precipitation", FeatureData::Int(vec![15])),
                ("target", FeatureData::Int(vec![0])),
            ],
        )?,
        DataCollection::from_slices(
            &[] as &[NoGeometry],
            &[TimeInterval::default(); 1],
            &[
                ("temp", FeatureData::Int(vec![8])),
                ("precipitation", FeatureData::Int(vec![16])),
                ("target", FeatureData::Int(vec![0])),
            ],
        )?,
    ])
    .boxed())
}

pub fn generate_raster_test_data_band_helper(
    v: Vec<i32>,
) -> Result<SourceOperator<MockRasterSourceParams<i32>>> {
    let tile_size_in_pixels = [4, 2].into();

    Ok(MockRasterSource {
        params: MockRasterSourceParams {
            data: vec![RasterTile2D::new_with_tile_info(
                TimeInterval::default(),
                TileInformation {
                    global_geo_transform: TestDefault::test_default(),
                    global_tile_position: [0, 0].into(),
                    tile_size_in_pixels,
                },
                MaskedGrid2D::new(
                    Grid2D::new(tile_size_in_pixels, v)?,
                    Grid2D::new(
                        tile_size_in_pixels,
                        vec![true, true, true, true, true, true, true, true],
                    )?,
                )?
                .into(),
            )],
            result_descriptor: RasterResultDescriptor {
                data_type: RasterDataType::U8,
                spatial_reference: SpatialReference::epsg_4326().into(),
                measurement: Measurement::Unitless,
                time: None,
                bbox: None,
                resolution: None,
            },
        },
    })
}
