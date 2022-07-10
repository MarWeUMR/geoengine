use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};
use geoengine_datatypes::hashmap;
use geoengine_datatypes::primitives::{Measurement, RasterQueryRectangle, SpatialPartition2D};
use geoengine_datatypes::raster::{
    EmptyGrid, Grid2D, GridShapeAccess, GridSize, NoDataValue, Pixel, RasterDataType,
    RasterPropertiesKey, RasterTile2D,
};
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};

use crate::engine::{
    ExecutionContext, InitializedRasterOperator, Operator, QueryContext, QueryProcessor,
    RasterOperator, RasterQueryProcessor, RasterResultDescriptor, SingleRasterSource,
    TypedRasterQueryProcessor,
};
use crate::error::Error;
use crate::util::Result;

use RasterDataType::F32 as RasterOut;

use super::{new_offset_key, new_slope_key};
use TypedRasterQueryProcessor::F32 as QueryProcessorOut;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct XgboostParams {}

pub type XgboostOperator = Operator<XgboostParams, SingleRasterSource>;

pub struct InitializedXgboostOperator {
    result_descriptor: RasterResultDescriptor,
    source: Box<dyn InitializedRasterOperator>,
}

type PixelOut = f32;
const OUT_NO_DATA_VALUE: PixelOut = PixelOut::NAN;

#[typetag::serde]
#[async_trait]
impl RasterOperator for XgboostOperator {
    async fn initialize(
        self: Box<Self>,
        context: &dyn ExecutionContext,
    ) -> Result<Box<dyn InitializedRasterOperator>> {
        let self_source = self.sources.clone();
        let raster = self_source.raster;
        let init_raster = raster.initialize(context).await.unwrap();

        let input = self.sources.raster.initialize(context).await?;
        let in_desc = input.result_descriptor();

        let out_desc = RasterResultDescriptor {
            spatial_reference: in_desc.spatial_reference,
            data_type: RasterOut,
            measurement: Measurement::Classification {
                measurement: "raw".into(),
                classes: hashmap!(0 => "Water Bodies".to_string()),
            },
            no_data_value: Some(f64::from(OUT_NO_DATA_VALUE)),
        };

        let initialized_operator = InitializedXgboostOperator {
            result_descriptor: out_desc,
            source: input,
        };

        Ok(initialized_operator.boxed())
    }
}

impl InitializedRasterOperator for InitializedXgboostOperator {
    fn result_descriptor(&self) -> &RasterResultDescriptor {
        &self.result_descriptor
    }

    fn query_processor(&self) -> Result<TypedRasterQueryProcessor> {
        let q = self.source.query_processor()?;

        Ok(match q {
            TypedRasterQueryProcessor::U8(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::U16(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::U32(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::U64(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::I8(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::I16(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::I32(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::I64(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::F32(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
            TypedRasterQueryProcessor::F64(p) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(p)))
            }
        })
    }
}

struct XgboostProcessor<Q, P>
where
    Q: RasterQueryProcessor<RasterType = P>,
{
    source: Q,
    offset_key: RasterPropertiesKey,
    slope_key: RasterPropertiesKey,
}

impl<Q, P> XgboostProcessor<Q, P>
where
    Q: RasterQueryProcessor<RasterType = P>,
    P: Pixel,
{
    pub fn new(source: Q) -> Self {
        Self {
            source,
            offset_key: new_offset_key(),
            slope_key: new_slope_key(),
        }
    }

    async fn process_tile_async(
        &self,
        tile: RasterTile2D<P>,
        pool: Arc<ThreadPool>,
    ) -> Result<RasterTile2D<PixelOut>> {
        if tile.is_empty() {
            return Ok(RasterTile2D::new_with_properties(
                tile.time,
                tile.tile_position,
                tile.global_geo_transform,
                EmptyGrid::new(tile.grid_array.grid_shape(), OUT_NO_DATA_VALUE).into(),
                tile.properties,
            ));
        }

        let offset = tile.properties.number_property::<f32>(&self.offset_key)?;
        let slope = tile.properties.number_property::<f32>(&self.slope_key)?;
        let mat_tile = tile.into_materialized_tile(); // NOTE: the tile is already materialized.

        let rad_grid = crate::util::spawn_blocking(move || {
            process_tile(&mat_tile.grid_array, offset, slope, &pool)
        })
        .await?;

        Ok(RasterTile2D::new_with_properties(
            mat_tile.time,
            mat_tile.tile_position,
            mat_tile.global_geo_transform,
            rad_grid.into(),
            mat_tile.properties,
        ))
    }
}

#[async_trait]
impl<Q, P> QueryProcessor for XgboostProcessor<Q, P>
where
    Q: QueryProcessor<Output = RasterTile2D<P>, SpatialBounds = SpatialPartition2D>,
    P: Pixel,
{
    type Output = RasterTile2D<PixelOut>;
    type SpatialBounds = SpatialPartition2D;

    async fn query<'a>(
        &'a self,
        query: RasterQueryRectangle,
        ctx: &'a dyn QueryContext,
    ) -> Result<BoxStream<'a, Result<Self::Output>>> {
        let src = self.source.query(query, ctx).await?;
        let rs = src.and_then(move |tile| self.process_tile_async(tile, ctx.thread_pool().clone()));
        Ok(rs.boxed())
    }
}

fn process_tile<P: Pixel>(
    grid: &Grid2D<P>,
    offset: f32,
    slope: f32,
    pool: &ThreadPool,
) -> Grid2D<PixelOut> {
    pool.install(|| {
        let rad_array = grid
            .data
            .par_chunks(grid.axis_size_x())
            .map(|row| {
                row.iter().map(|p| {
                    if grid.is_no_data(*p) {
                        OUT_NO_DATA_VALUE
                    } else {
                        let val: PixelOut = (p).as_();
                        offset + val * slope
                    }
                })
            })
            .flatten_iter()
            .collect::<Vec<PixelOut>>();

        Grid2D::new(grid.grid_shape(), rad_array, Some(OUT_NO_DATA_VALUE))
            .expect("raster creation must succeed")
    })
}

#[cfg(test)]
mod tests {
    use crate::engine::{
        MockExecutionContext, MockQueryContext, QueryProcessor, RasterOperator,
        RasterQueryProcessor, RasterResultDescriptor, SingleRasterSource,
    };
    use crate::mock::{MockRasterSource, MockRasterSourceParams};
    use crate::processing::meteosat::xgboost::{XgboostOperator, XgboostParams};
    use crate::processing::meteosat::{
        new_channel_key, new_offset_key, new_satellite_key, new_slope_key, test_util,
    };
    use crate::source::{
        FileNotFoundHandling, GdalDatasetGeoTransform, GdalDatasetParameters, GdalMetaDataRegular,
        GdalMetadataMapping, GdalSource, GdalSourceParameters, GdalSourceTimePlaceholder,
        TimeReference,
    };
    use crate::util::Result;
    use futures::StreamExt;
    use geoengine_datatypes::dataset::{DatasetId, InternalDatasetId};
    use geoengine_datatypes::primitives::{
        Coordinate2D, Measurement, QueryRectangle, RasterQueryRectangle, SpatialPartition2D,
        SpatialResolution, TimeGranularity, TimeInstance, TimeInterval, TimeStep,
    };
    use geoengine_datatypes::raster::{
        Grid2D, GridOrEmpty, RasterDataType, RasterPropertiesEntryType, TilingSpecification,
    };
    use geoengine_datatypes::spatial_reference::{
        SpatialReference, SpatialReferenceAuthority, SpatialReferenceOption,
    };
    use geoengine_datatypes::util::test::TestDefault;
    use geoengine_datatypes::util::Identifier;
    use geoengine_datatypes::{hashmap, test_data};
    use num_traits::AsPrimitive;

    use std::path::PathBuf;

    fn get_gdal_config_metadata(path: &str) -> GdalMetaDataRegular {
        let no_data_value = Some(-1000.0);

        let gdal_config_metadata = GdalMetaDataRegular {
            result_descriptor: RasterResultDescriptor {
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::SpatialReference(SpatialReference::new(
                    SpatialReferenceAuthority::Epsg,
                    32632,
                )),
                measurement: Measurement::Classification {
                    measurement: "raw".into(),
                    classes: hashmap!(0 => "Water Bodies".to_string()),
                },
                no_data_value,
            },
            params: GdalDatasetParameters {
                file_path: PathBuf::from(test_data!(path)),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform {
                    origin_coordinate: (474112.0, 5646336.0).into(),
                    x_pixel_size: 10.0,
                    y_pixel_size: -10.0,
                },
                width: 4864,
                height: 3431,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![
                    GdalMetadataMapping::identity(
                        new_offset_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                    GdalMetadataMapping::identity(
                        new_slope_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                    GdalMetadataMapping::identity(
                        new_channel_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                    GdalMetadataMapping::identity(
                        new_satellite_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                ]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_087___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep {
                granularity: TimeGranularity::Minutes,
                step: 15,
            },
        };

        gdal_config_metadata
    }

    async fn initialize_operator(
        gcm: GdalMetaDataRegular,
        tile_size: usize,
    ) -> Box<dyn RasterQueryProcessor<RasterType = i16>> {
        let tiling_specification =
            TilingSpecification::new(Coordinate2D::default(), [tile_size, tile_size].into());

        let mut mc = MockExecutionContext::test_default();
        mc.tiling_specification = tiling_specification;

        let id = DatasetId::Internal {
            dataset_id: InternalDatasetId::new(),
        };

        let op = Box::new(GdalSource {
            params: GdalSourceParameters {
                dataset: id.clone(),
            },
        });

        // let gcm = get_gdal_config_metadata();

        mc.add_meta_data(id, Box::new(gcm.clone()));

        let rqp_gt = op
            .initialize(&mc)
            .await
            .unwrap()
            .query_processor()
            .unwrap()
            .get_i16()
            .unwrap();

        rqp_gt
    }

    async fn get_band_data(
        rqp_gt: Box<dyn RasterQueryProcessor<RasterType = i16>>,
    ) -> Vec<
        geoengine_datatypes::raster::BaseTile<
            GridOrEmpty<geoengine_datatypes::raster::GridShape<[usize; 2]>, i16>,
        >,
    > {
        let ctx = MockQueryContext::test_default();

        let query_bbox = SpatialPartition2D::new(
            (474112.000, 5646336.000).into(),
            (522752.000, 5612026.000).into(),
        )
        .unwrap();
        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = QueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1590969600000, 1590969600000).unwrap(),
            spatial_resolution: query_spatial_resolution,
        };

        let mut stream = rqp_gt
            .raster_query(qry_rectangle.clone(), &ctx)
            .await
            .unwrap();

        let mut tile_buffer: Vec<_> = Vec::new();

        while let Some(processor) = stream.next().await {
            tile_buffer.push(processor.unwrap());
        }

        tile_buffer
    }

    async fn get_src(
        gcm: GdalMetaDataRegular,
    ) -> crate::engine::SourceOperator<MockRasterSourceParams<i16>> {
        //let path = "s2_10m_de_marburg/target.tiff";
        let init_op = initialize_operator(gcm, 512).await;
        let tile_vec = get_band_data(init_op).await;

        println!("n tiles: {:?}", tile_vec.len());

        let measurement = Measurement::Classification {
            measurement: "whyyy".into(),
            classes: hashmap!(0 => "Water Bodies".to_string()),
        };
        let mrs = MockRasterSource {
            params: MockRasterSourceParams {
                data: tile_vec,
                result_descriptor: RasterResultDescriptor {
                    data_type: RasterDataType::I16,
                    spatial_reference: SpatialReference::new(
                        SpatialReferenceAuthority::Epsg,
                        32632,
                    )
                    .into(),
                    measurement: measurement,
                    no_data_value: Some(-1000.0).map(AsPrimitive::as_),
                },
            },
        };
        mrs
    }

    #[tokio::test]
    async fn xg_op_test() {
        // !----------------------------------------------------
        // TODO: irgendwie die mock raster source implementieren
        // !----------------------------------------------------
        let path = "s2_10m_de_marburg/target.tiff";
        let gcm = get_gdal_config_metadata(path);

        let tiling_specification =
            TilingSpecification::new(Coordinate2D::default(), [512, 512].into());
        let id = DatasetId::Internal {
            dataset_id: InternalDatasetId::new(),
        };
        let mut ctx = MockExecutionContext::test_default();
        ctx.tiling_specification = tiling_specification;

        ctx.add_meta_data(id.clone(), Box::new(gcm.clone()));

        let query_bbox = SpatialPartition2D::new(
            (474112.000, 5646336.000).into(),
            (522752.000, 5612026.000).into(),
        )
        .unwrap();
        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = RasterQueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1590969600000, 1590969600000).unwrap(),
            spatial_resolution: query_spatial_resolution,
        };

        let src_operator = get_src(gcm).await;

        let xg = XgboostOperator {
            params: XgboostParams {},
            sources: SingleRasterSource {
                raster: src_operator.boxed(),
            },
        };

        let closure = || RasterOperator::boxed(xg);
        let result = test_util::process(closure, qry_rectangle, &ctx).await;

        println!("done");
        let r = result.unwrap();
        println!("{:?}", r);
    }
}
