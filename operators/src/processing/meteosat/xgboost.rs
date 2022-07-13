use std::pin::Pin;
use std::sync::Arc;
use std::time::SystemTime;
use std::{mem, path};

use crate::error;
use crate::util::stream_zip::StreamVectorZip;
use async_trait::async_trait;
use futures::stream::{self, select_all, BoxStream};
use futures::{future, Stream, StreamExt, TryFutureExt, TryStreamExt};
use geoengine_datatypes::hashmap;
use geoengine_datatypes::primitives::{Measurement, RasterQueryRectangle, SpatialPartition2D};
use geoengine_datatypes::raster::{
    BaseTile, ConvertDataTypeParallel, EmptyGrid, Grid, Grid2D, GridOrEmpty, GridShape,
    GridShapeAccess, GridSize, NoDataValue, Pixel, RasterDataType, RasterPropertiesKey,
    RasterTile2D,
};
use ndarray::Array2;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use snafu::ensure;
use xgboost_bindings::{Booster, DMatrix};

use crate::engine::{
    ExecutionContext, InitializedRasterOperator, MultipleRasterSources, Operator, QueryContext,
    QueryProcessor, RasterOperator, RasterQueryProcessor, RasterResultDescriptor,
    SingleRasterSource, TypedRasterQueryProcessor,
};
use crate::error::Error;
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
            QueryProcessorOut(p) => QueryProcessorOut(Box::new(XgboostProcessor::new(vec_of_rqps))),
            TypedRasterQueryProcessor::U8(_) => todo!(),
            TypedRasterQueryProcessor::U16(_) => todo!(),
            TypedRasterQueryProcessor::U32(_) => todo!(),
            TypedRasterQueryProcessor::U64(_) => todo!(),
            TypedRasterQueryProcessor::I8(_) => todo!(),
            TypedRasterQueryProcessor::I16(_) => {
                QueryProcessorOut(Box::new(XgboostProcessor::new(vec_of_rqps)))
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
}

impl<Q, P> XgboostProcessor<Q, P>
where
    Q: RasterQueryProcessor<RasterType = P>,
    P: Pixel,
{
    pub fn new(sources: Vec<Q>) -> Self {
        Self { sources }
    }

    async fn process_tile_async2(
        &self,
        tiles: Vec<Vec<P>>,
        pool: Arc<ThreadPool>,
    ) -> Result<Vec<f32>, xgboost_bindings::XGBError> {
        let n_cols = tiles.len();
        let n_rows = tiles.first().unwrap().len();

        let mut data: Vec<f64> = Vec::new();
        for i in 0..n_rows {
            let mut row = Vec::new();
            for c in 0..n_cols {
                let val = tiles.get(c).unwrap().get(i).unwrap();
                let v = (val).as_();
                row.push(v);
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

        let booster_model = Booster::load(path::Path::new("model.json")).unwrap();
        let start = SystemTime::now();

        let mut out_dim: u64 = 0;
        let shp = &[n_rows as u64, n_cols as u64];
        let result = booster_model.predict_from_dmat(&xg_matrix, shp, &mut out_dim);
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("it took {} ms", duration.as_millis());

        result
    }

    // apply xgboost model prediction here
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
        let grid_shape = tile.grid_shape();

        let mat_tile = tile.into_materialized_tile(); // NOTE: the tile is already materialized.

        let data_of_tile = mat_tile.grid_array.data;
        let predicted_tile_data = self.predict(data_of_tile).unwrap();
        let no_data = -1000.0;
        let predicted_grid = Grid2D::new(grid_shape, predicted_tile_data, Some(no_data))
            .expect("raster creation must succeed");

        Ok(RasterTile2D::new_with_properties(
            mat_tile.time,
            mat_tile.tile_position,
            mat_tile.global_geo_transform,
            predicted_grid.into(),
            mat_tile.properties,
        ))
    }

    fn predict(&self, tile: Vec<P>) -> Result<Vec<f32>, xgboost_bindings::XGBError> {
        // make xg compatible, trainable datastructure
        let xg_matrix = self.make_xg_data_no_labels(tile);

        let booster_model = Booster::load(path::Path::new("model.json"))?;
        let start = SystemTime::now();

        let mut out_dim: u64 = 0;
        let shp = &[(512 as u64 * 512 as u64), 1];
        let result = booster_model.predict_from_dmat(&xg_matrix, shp, &mut out_dim);
        let end = SystemTime::now();
        let duration = end.duration_since(start).unwrap();
        println!("it took {} ms", duration.as_millis());
        result
    }

    fn make_xg_data_no_labels(&self, tile: Vec<P>) -> DMatrix {
        let mut tabular_like_data_vec = Vec::new();

        for i in 0..tile.len() {
            let e1 = tile.get(i).unwrap();

            let row = vec![e1.to_owned().as_()];
            tabular_like_data_vec.extend_from_slice(&row);
        }

        let n_cols = 1;
        let n_rows = tabular_like_data_vec.len() / n_cols;

        let data_arr_2d =
            Array2::from_shape_vec((n_rows as usize, n_cols), tabular_like_data_vec).unwrap();

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
            tile.len(),
            n_cols as usize,
        )
        .unwrap();

        // set labels
        // TODO: make more generic

        xg_matrix
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
        let mut buffer = Vec::new();

        for band in self.sources.iter() {
            let mut stream = band.query(query, ctx).await?;

            buffer.push(stream);
        }

        let s = StreamVectorZip::new(buffer);

        let stream_of_tiled_bands = s
            .then(|elem| async move {
                let extracted_data_from_bands_in_tile: Vec<Vec<P>> = elem
                    .into_iter()
                    .map(|t| {
                        let tile = t.unwrap();
                        let mat_tile = tile.into_materialized_tile();
                        let data = mat_tile.grid_array.data;

                        data
                    })
                    .collect();
                extracted_data_from_bands_in_tile
            })
            .boxed();

        let x = self.sources.first().unwrap();
        let mut xx = x.query(query, ctx).await.unwrap();
        let tile_ref = xx.next().await.unwrap().unwrap();

        // let mut tiled_bands: Vec<_> = s.collect().await;
        // let tile_vec = tiled_bands.first().unwrap();
        // let tile_ref = tile_vec.first().unwrap().as_ref().unwrap();

        let grid_shp = tile_ref.grid_shape();
        let time = query.time_interval;
        let tile_position = tile_ref.tile_position;
        let global_geo_transform = tile_ref.global_geo_transform;
        let properties = tile_ref.properties.clone();

        let y: Vec<_> = stream_of_tiled_bands.collect().await;
        let predicted_data = predict_tile_data(y).await;

        println!("generating new tiles");
        let predicted_tiles: Vec<_> = predicted_data
            .into_iter()
            .map(|tile| {
                let no_data = -1000.0;
                let predicted_grid = Grid2D::new(grid_shp, tile, Some(no_data))
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
            })
            .collect::<Vec<Result<_>>>();

        let whats_in_the_box = Box::pin(futures::stream::iter(predicted_tiles));
        let aaa = whats_in_the_box.boxed();

        let src1 = self.sources.get(0).unwrap().query(query, ctx).await?;
        let src2 = self.sources.get(1).unwrap().query(query, ctx).await?;

        // let src = self.sources.query(query, ctx).await?;
        let rs =
            src1.and_then(move |tile| self.process_tile_async(tile, ctx.thread_pool().clone()));
        Ok(aaa)
    }
}

async fn predict_tile_data<P: Pixel>(stream: Vec<Vec<Vec<P>>>) -> Vec<Vec<f32>> {
    // let stream: Vec<_> = stream_of_tiled_bands.collect().await;

    let x: Vec<_> = stream
        .iter()
        .enumerate()
        .map(|(i, elem)| {
            println!("predicting tile: {:?} now", i);
            let n_cols = elem.len();
            let n_rows = elem.first().unwrap().len();

            let mut data: Vec<f64> = Vec::new();
            for i in 0..n_rows {
                let mut row = Vec::new();
                for c in 0..n_cols {
                    let val = elem.get(c).unwrap().get(i).unwrap();
                    let v = (val).as_();
                    row.push(v);
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

            let booster_model =
                Booster::load(path::Path::new("/workspace/geoengine/operators/model.json"))
                    .unwrap();
            let start = SystemTime::now();

            let mut out_dim: u64 = 0;
            let shp = &[n_rows as u64, n_cols as u64];
            let result = booster_model.predict_from_dmat(&xg_matrix, shp, &mut out_dim);
            let end = SystemTime::now();
            let duration = end.duration_since(start).unwrap();
            println!("prediction from xg took {} ms", duration.as_millis());

            result.unwrap()
        })
        .collect();

    x
}

fn process_tile<P: Pixel>(
    grid: &Grid2D<P>,
    offset: f32,
    slope: f32,
    pool: &ThreadPool,
) -> Grid2D<PixelOut> {
    println!("process_tile");
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
        MockExecutionContext, MockQueryContext, MultipleRasterSources, QueryProcessor,
        RasterOperator, RasterQueryProcessor, RasterResultDescriptor, SingleRasterSource,
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

    async fn get_src(gcm_vec: Vec<GdalMetaDataRegular>) -> Vec<Box<dyn RasterOperator>> {
        let mut src_vec = Vec::new();

        for gcm in gcm_vec.into_iter() {
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
            src_vec.push(mrs.boxed());
        }
        src_vec
    }

    #[tokio::test]
    async fn xg_op_test() {
        // !----------------------------------------------------
        // TODO: xgboost predictions in xgprocessor einflechten
        // !----------------------------------------------------
        // let path = "s2_10m_de_marburg/target.tiff";
        let paths = vec![
            "s2_10m_de_marburg/b02.tiff",
            "s2_10m_de_marburg/b03.tiff",
            "s2_10m_de_marburg/b04.tiff",
            "s2_10m_de_marburg/b08.tiff",
        ];
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
