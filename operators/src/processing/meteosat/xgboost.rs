use std::sync::Arc;
use std::time::SystemTime;
use std::{mem, path};

use crate::error::Error;
use crate::util::stream_zip::StreamVectorZip;
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::{future, StreamExt};
use geoengine_datatypes::hashmap;
use geoengine_datatypes::primitives::{
    Measurement, RasterQueryRectangle, SpatialPartition2D, TimeInterval,
};
use geoengine_datatypes::raster::{
    BaseTile, GeoTransform, Grid2D, GridIdx, GridOrEmpty, GridShape, GridShapeAccess, Pixel,
    RasterDataType, RasterProperties, RasterTile2D,
};
use ndarray::Array2;
use rayon::iter::IntoParallelIterator;
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use xgboost_bindings::{Booster, DMatrix};

use crate::engine::{
    ExecutionContext, InitializedRasterOperator, MultipleRasterSources, Operator, QueryContext,
    QueryProcessor, RasterOperator, RasterQueryProcessor, RasterResultDescriptor,
    TypedRasterQueryProcessor,
};
use crate::util::Result;

use RasterDataType::F32 as RasterOut;

use TypedRasterQueryProcessor::F32 as QueryProcessorOut;

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(rename_all = "camelCase")]
pub struct XgboostParams {}

pub type XgboostOperator = Operator<XgboostParams, MultipleRasterSources>;

pub struct InitializedXgboostOperator {
    result_descriptor: RasterResultDescriptor,
    sources: Vec<Box<dyn InitializedRasterOperator>>,
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
        println!("initializing raster sources");
        let self_source = self.sources.clone();
        let rasters = self_source.rasters;

        let init_rasters = future::try_join_all(
            rasters
                .iter()
                .map(|raster| raster.clone().initialize(context)),
        )
        .await
        .unwrap();

        let input = init_rasters.get(0).unwrap().clone();
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
            sources: init_rasters,
        };

        Ok(initialized_operator.boxed())
    }
}

impl InitializedRasterOperator for InitializedXgboostOperator {
    fn result_descriptor(&self) -> &RasterResultDescriptor {
        &self.result_descriptor
    }

    fn query_processor(&self) -> Result<TypedRasterQueryProcessor> {
        println!("typing rasters");
        let vec_of_rqps = self
            .sources
            .iter()
            .map(|init_raster| {
                let typed_raster_processor = init_raster.query_processor().unwrap();
                let boxed_raster_data = typed_raster_processor.get_i16().unwrap();
                boxed_raster_data
            })
            .collect();

        let typed_rqp = match self.sources.first().unwrap().query_processor().unwrap() {
            QueryProcessorOut(_p) => QueryProcessorOut(Box::new(XgboostProcessor::new(
                vec_of_rqps,
                "/workspace/geoengine/operators/model.json".into(),
            ))),
            TypedRasterQueryProcessor::U8(_) => todo!(),
            TypedRasterQueryProcessor::U16(_) => todo!(),
            TypedRasterQueryProcessor::U32(_) => todo!(),
            TypedRasterQueryProcessor::U64(_) => todo!(),
            TypedRasterQueryProcessor::I8(_) => todo!(),
            TypedRasterQueryProcessor::I16(_) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(
                    vec_of_rqps,
                    "/workspace/geoengine/operators/model.json".into(),
                )))
            }
            TypedRasterQueryProcessor::I32(_) => todo!(),
            TypedRasterQueryProcessor::I64(_) => todo!(),
            TypedRasterQueryProcessor::F64(_) => todo!(),
        };
        Ok(typed_rqp)
    }
}

struct XgboostProcessor<Q, P>
where
    Q: RasterQueryProcessor<RasterType = P>,
{
    sources: Vec<Q>,
    model_file: String,
    bst: Booster,
}

impl<Q, P> XgboostProcessor<Q, P>
where
    Q: RasterQueryProcessor<RasterType = P>,
    P: Pixel,
{
    pub fn new(sources: Vec<Q>, model_file_path: String) -> Self {
        Self {
            sources,
            model_file: model_file_path.clone(),
            bst: Booster::load(model_file_path.clone()).unwrap(),
        }
    }

    async fn predict_tile_data_async(
        &self,
        tile: Vec<Result<BaseTile<GridOrEmpty<GridShape<[usize; 2]>, P>>, Error>>,
        grid_shape: GridShape<[usize; 2]>,
        time: TimeInterval,
        tile_position: GridIdx<[isize; 2]>,
        global_geo_transform: GeoTransform,
        properties: RasterProperties,
        ctx: &dyn QueryContext,
        pool: Arc<ThreadPool>,
    ) -> Result<RasterTile2D<PixelOut>> {
        let bands_of_tile: Vec<_> = tile
            .into_iter()
            .map(|band| {
                let b = band.unwrap();
                let mat = b.into_materialized_tile();
                mat.grid_array.data
            })
            .collect();

        let n_cols = bands_of_tile.len();
        let n_rows = bands_of_tile.first().unwrap().len();

        let xg_matrix = crate::util::spawn_blocking(move || {
            prepare_xgmatrix(bands_of_tile, n_cols, n_rows, &pool)
        })
        .await
        .unwrap();

        let start = SystemTime::now();

        let mut out_dim: u64 = 0;
        let shp = &[n_rows as u64, n_cols as u64];
        let result = self.bst.predict_from_dmat(&xg_matrix, shp, &mut out_dim);
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("prediction from xg took {} ms", duration.as_millis());

        let no_data = Some(-1000.0);
        let predicted_grid = Grid2D::new(grid_shape, result.unwrap(), no_data)
            .expect("raster creation must succeed");

        let rt: BaseTile<GridOrEmpty<GridShape<[usize; 2]>, f32>> =
            RasterTile2D::new_with_properties(
                time,
                tile_position,
                global_geo_transform,
                predicted_grid.into(),
                properties.clone(),
            );

        Ok(rt)
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
        println!("querying");
        let mut band_buffer = Vec::new();

        for band in self.sources.iter() {
            let stream = band.query(query, ctx).await?;
            band_buffer.push(stream);
        }

        let zipped_stream_tiled_bands = StreamVectorZip::new(band_buffer);

        // TODO: better solution for this?
        // we need to get meta data somehow
        let source = self.sources.first().unwrap();
        let mut source_stream = source.query(query, ctx).await.unwrap();
        let tile = source_stream.next().await.unwrap().unwrap();

        let grid_shp = tile.grid_shape();
        let time = query.time_interval;
        let tile_position = tile.tile_position;
        let global_geo_transform = tile.global_geo_transform;
        let properties = tile.properties.clone();

        // test
        let rs = zipped_stream_tiled_bands.then(move |tile| {
            self.predict_tile_data_async(
                tile,
                grid_shp,
                time,
                tile_position,
                global_geo_transform,
                properties.clone(),
                ctx,
                ctx.thread_pool().clone(),
            )
        });
        Ok(rs.boxed())
        // let rss = rs.boxed();
        // /test

        // let stream_of_band_data_per_tile = zipped_stream_tiled_bands
        //     .then(|bands| async move {
        //         let extracted_data_from_bands_in_tile: Vec<Vec<P>> = bands
        //             .into_iter()
        //             .map(|base_tile_band| {
        //                 let tile = base_tile_band.unwrap();
        //                 let mat_tile = tile.into_materialized_tile();
        //                 let data = mat_tile.grid_array.data;

        //                 data
        //             })
        //             .collect();
        //         extracted_data_from_bands_in_tile
        //     })
        //     .boxed();

        // // TODO: better solution for this?
        // // we need to get meta data somehow
        // let source = self.sources.first().unwrap();
        // let mut source_stream = source.query(query, ctx).await.unwrap();
        // let tile = source_stream.next().await.unwrap().unwrap();

        // let grid_shp = tile.grid_shape();
        // let time = query.time_interval;
        // let tile_position = tile.tile_position;
        // let global_geo_transform = tile.global_geo_transform;
        // let properties = tile.properties.clone();

        // let vec_of_band_data_per_tile: Vec<_> = stream_of_band_data_per_tile.collect().await;
        // let bst = &self.bst;
        // let predicted_data = predict_tile_data(vec_of_band_data_per_tile, bst).await;

        // println!("generating new tiles");
        // let predicted_tiles: Vec<_> = predicted_data
        //     .into_iter()
        //     .map(|tile_data| {
        //         let no_data = -1000.0;
        //         let predicted_grid = Grid2D::new(grid_shp, tile_data, Some(no_data))
        //             .expect("raster creation must succeed");

        //         let rt: BaseTile<GridOrEmpty<GridShape<[usize; 2]>, f32>> =
        //             RasterTile2D::new_with_properties(
        //                 time,
        //                 tile_position,
        //                 global_geo_transform,
        //                 predicted_grid.into(),
        //                 properties.clone(),
        //             );

        //         Ok(rt)
        //     })
        //     .collect::<Vec<Result<_>>>();

        // let whats_in_the_box = Box::pin(futures::stream::iter(predicted_tiles));
        // let aaa = whats_in_the_box.boxed();

        // Ok(aaa)
    }
}

fn prepare_xgmatrix<P: Pixel>(
    bands_of_tile: Vec<Vec<P>>,
    n_cols: usize,
    n_rows: usize,
    pool: &ThreadPool,
) -> DMatrix {
    let matrix = pool.install(|| -> DMatrix {
        let mut data: Vec<f64> = Vec::new();
        for i in 0..n_rows {
            let mut row = Vec::new();
            for col in 0..n_cols {
                let pixel_value: f64 = bands_of_tile.get(col).unwrap().get(i).unwrap().as_();
                row.push(pixel_value);
            }

            data.extend_from_slice(&row);
        }

        let data_arr_2d = Array2::from_shape_vec((n_rows, n_cols), data).unwrap();

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let xg_matrix = DMatrix::from_col_major_f64(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            n_rows,
            n_cols,
        )
        .unwrap();
        xg_matrix
    });
    matrix
}
/// Takes in the bands per tile. Returns the predicted data per tile.
/// The method takes appropriate actions to turn the data into xgboost compatible matrix data.
async fn predict_tile_data<P: Pixel>(stream: Vec<Vec<Vec<P>>>, bst: &Booster) -> Vec<Vec<f32>> {
    let input_data_stream: Vec<_> = stream
        .iter()
        .enumerate()
        .map(|(i, bands_of_tile)| {
            println!("predicting tile: {:?} now", i);
            let n_cols = bands_of_tile.len();
            let n_rows = bands_of_tile.first().unwrap().len();

            let mut data: Vec<f64> = Vec::new();
            for i in 0..n_rows {
                let mut row = Vec::new();
                for col in 0..n_cols {
                    let pixel_value: f64 = bands_of_tile.get(col).unwrap().get(i).unwrap().as_();
                    row.push(pixel_value);
                }

                data.extend_from_slice(&row);
            }

            let data_arr_2d = Array2::from_shape_vec((n_rows, n_cols), data).unwrap();

            // define information needed for xgboost
            let strides_ax_0 = data_arr_2d.strides()[0] as usize;
            let strides_ax_1 = data_arr_2d.strides()[1] as usize;
            let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
            let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

            // get xgboost style matrices
            let xg_matrix = DMatrix::from_col_major_f64(
                data_arr_2d.as_slice_memory_order().unwrap(),
                byte_size_ax_0,
                byte_size_ax_1,
                n_rows,
                n_cols,
            )
            .unwrap();

            let start = SystemTime::now();

            let mut out_dim: u64 = 0;
            let shp = &[n_rows as u64, n_cols as u64];
            let result = bst.predict_from_dmat(&xg_matrix, shp, &mut out_dim);
            let end = SystemTime::now();
            let duration = end.duration_since(start).unwrap();
            println!("prediction from xg took {} ms", duration.as_millis());

            result.unwrap()
        })
        .collect();

    input_data_stream
}

#[cfg(test)]
mod tests {
    use crate::engine::{
        MockExecutionContext, MockQueryContext, MultipleRasterSources, RasterOperator,
        RasterQueryProcessor, RasterResultDescriptor,
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
    use futures::StreamExt;
    use geoengine_datatypes::dataset::{DatasetId, InternalDatasetId};
    use geoengine_datatypes::primitives::{
        Coordinate2D, Measurement, QueryRectangle, RasterQueryRectangle, SpatialPartition2D,
        SpatialResolution, TimeGranularity, TimeInstance, TimeInterval, TimeStep,
    };
    use geoengine_datatypes::raster::{
        GridOrEmpty, RasterDataType, RasterPropertiesEntryType, TilingSpecification,
    };
    use geoengine_datatypes::spatial_reference::{
        SpatialReference, SpatialReferenceAuthority, SpatialReferenceOption,
    };
    use geoengine_datatypes::util::test::TestDefault;
    use geoengine_datatypes::util::Identifier;
    use geoengine_datatypes::{hashmap, test_data};
    use num_traits::AsPrimitive;

    use std::path::PathBuf;

    fn get_gdal_config_metadata(paths: Vec<&str>) -> Vec<GdalMetaDataRegular> {
        let no_data_value = Some(-1000.0);

        let mut gmdr = Vec::new();

        for path in paths.iter() {
            let gdal_config_metadata = GdalMetaDataRegular {
                result_descriptor: RasterResultDescriptor {
                    data_type: RasterDataType::I16,
                    spatial_reference: SpatialReferenceOption::SpatialReference(
                        SpatialReference::new(SpatialReferenceAuthority::Epsg, 32632),
                    ),
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
            gmdr.push(gdal_config_metadata);
        }

        gmdr
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

        let props = test_util::create_properties(None, None, Some(11.0), Some(2.0));

        while let Some(processor) = stream.next().await {
            let mut tile = processor.unwrap();
            tile.properties = props.clone();
            tile_buffer.push(tile);
        }

        tile_buffer
    }

    /// Build a MockRasterSource to use with the operator
    async fn get_src(gcm_vec: Vec<GdalMetaDataRegular>) -> Vec<Box<dyn RasterOperator>> {
        let mut src_vec = Vec::new();

        for gcm in gcm_vec.into_iter() {
            let init_op = initialize_operator(gcm, 512).await;
            let tile_vec = get_band_data(init_op).await;

            println!("n tiles: {:?}", tile_vec.len());

            let measurement = Measurement::Classification {
                measurement: "water".into(),
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
                        measurement,
                        no_data_value: Some(-1000.0).map(AsPrimitive::as_),
                    },
                },
            };
            src_vec.push(mrs.boxed());
        }
        src_vec
    }

    #[tokio::test]
    async fn xg_op_test() {
        // TODO: investigate prediction step performance -> parallelize possible?
        // setup data to predict
        let paths = vec![
            "s2_10m_de_marburg/b02.tiff",
            "s2_10m_de_marburg/b03.tiff",
            "s2_10m_de_marburg/b04.tiff",
            "s2_10m_de_marburg/b08.tiff",
        ];

        // setup context and meta data
        let gcm = get_gdal_config_metadata(paths);

        let tiling_specification =
            TilingSpecification::new(Coordinate2D::default(), [512, 512].into());

        let id = DatasetId::Internal {
            dataset_id: InternalDatasetId::new(),
        };

        let mut ctx = MockExecutionContext::test_default();

        ctx.tiling_specification = tiling_specification;

        ctx.add_meta_data(id.clone(), Box::new(gcm.first().unwrap().clone()));

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

        // this operator prepares the input data
        let srcs = get_src(gcm.clone()).await;

        // xg-operator takes the input data for further processing
        let xg = XgboostOperator {
            params: XgboostParams {},
            sources: MultipleRasterSources { rasters: srcs },
        };

        let closure = || RasterOperator::boxed(xg);
        let result = test_util::process(closure, qry_rectangle, &ctx).await;

        println!("done");
        let r = result.unwrap();
        println!("{:?}", r);
    }
}
