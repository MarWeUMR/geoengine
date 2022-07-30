use rayon::prelude::*;
use std::mem;
use std::sync::Arc;
use tokio::time::Instant;

use crate::util::stream_zip::StreamVectorZip;
use async_trait::async_trait;
use futures::stream;
use futures::stream::BoxStream;
use futures::{future, StreamExt};

use futures::TryStreamExt;
use geoengine_datatypes::hashmap;
use geoengine_datatypes::primitives::{Measurement, RasterQueryRectangle, SpatialPartition2D};
use geoengine_datatypes::raster::{
    BaseTile, Grid2D, GridOrEmpty, GridShape, GridShapeAccess, Pixel, RasterDataType, RasterTile2D,
};
use ndarray::Array2;

use rayon::iter::ParallelIterator;
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

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
pub struct XgboostParams {
    model_file_path: String,
}

pub type XgboostOperator = Operator<XgboostParams, MultipleRasterSources>;
pub type Tile<P> = BaseTile<GridOrEmpty<GridShape<[usize; 2]>, P>>;

pub struct InitializedXgboostOperator {
    result_descriptor: RasterResultDescriptor,
    sources: Vec<Box<dyn InitializedRasterOperator>>,
    model_file_path: String,
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

        let input = init_rasters.get(0).unwrap();
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
            model_file_path: self.params.model_file_path,
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
                self.model_file_path.clone(),
            ))),
            TypedRasterQueryProcessor::I16(_) => QueryProcessorOut(Box::new(
                XgboostProcessor::new(vec_of_rqps, self.model_file_path.clone()),
            )),
            _ => todo!(),
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
}

impl<Q, P> XgboostProcessor<Q, P>
where
    Q: RasterQueryProcessor<RasterType = P>,
    P: Pixel,
{
    pub fn new(sources: Vec<Q>, model_file_path: String) -> Self {
        Self {
            sources,
            model_file: std::fs::read_to_string(model_file_path).unwrap(),
        }
    }

    async fn process_tile_async(
        &self,
        bands_of_tile: Vec<Result<Tile<P>, crate::error::Error>>,
        pool: Arc<ThreadPool>,
    ) -> Result<RasterTile2D<PixelOut>> {
        println!("async_func");
        let t = bands_of_tile.first().unwrap();
        let tile = t.as_ref().unwrap().clone();
        let n_rows = tile.grid_shape_array()[0];
        let n_cols = tile.grid_shape_array()[1];
        let n_bands = bands_of_tile.len() as i32;
        let grid_shape = tile.grid_shape();
        let props = tile.properties;

        let rasters: Vec<Vec<f64>> = bands_of_tile
            .into_iter()
            .map(|band| {
                let band_ok = band.unwrap();
                let mat_tile = band_ok.into_materialized_tile();

                mat_tile
                    .grid_array
                    .data
                    .into_iter()
                    .map(num_traits::AsPrimitive::as_)
                    .collect::<Vec<f64>>()
            })
            .collect();

        let mut pixels: Vec<f64> = Vec::new();

        for row in 0..(n_rows * n_cols) {
            let mut row_data: Vec<f64> = Vec::new();
            for col in 0..n_bands {
                let pxl = rasters
                    .get(col as usize)
                    .unwrap()
                    .get(row)
                    .unwrap()
                    .to_owned();
                row_data.push(pxl);
            }
            pixels.extend_from_slice(&row_data);
        }

        let x = self.model_file.clone();
        let predicted_grid = crate::util::spawn_blocking(move || {
            process_tile(&pixels, &pool, x.as_bytes(), grid_shape, n_bands as usize)
        })
        .await
        .unwrap();

        let rt: BaseTile<GridOrEmpty<GridShape<[usize; 2]>, f32>> =
            RasterTile2D::new_with_properties(
                tile.time,
                tile.tile_position,
                tile.global_geo_transform,
                predicted_grid.into(),
                props.clone(),
            );

        Ok(rt)
    }
}

fn process_tile(
    bands_of_tile: &Vec<f64>,
    pool: &ThreadPool,
    model_file: &[u8],
    grid_shape: GridShape<[usize; 2]>,
    n_bands: usize,
) -> geoengine_datatypes::raster::Grid<GridShape<[usize; 2]>, f32> {
    pool.install(|| {
        println!(
            "processing tile with thread {:?}",
            pool.current_thread_index()
        );
        let chunk_size = 64 * 64;
        let res: Vec<_> = bands_of_tile
            .par_chunks(n_bands * chunk_size)
            .map(|elem| {
                let v = Vec::from(elem);
                let true_chunk_size = v.len() / n_bands;
                println!("chunk size: {:?}", v.len());
                let data_arr_2d = Array2::from_shape_vec((true_chunk_size, n_bands), v).unwrap();
                let strides_ax_0 = data_arr_2d.strides()[0] as usize;
                let strides_ax_1 = data_arr_2d.strides()[1] as usize;
                let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
                let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

                // get xgboost style matrices
                let xg_matrix = DMatrix::from_col_major_f64(
                    data_arr_2d.as_slice_memory_order().unwrap(),
                    byte_size_ax_0,
                    byte_size_ax_1,
                    true_chunk_size,
                    n_bands,
                )
                .unwrap();

                let mut out_dim: u64 = 0;
                let shp = &[true_chunk_size as u64, n_bands as u64];
                let bst = Booster::load_buffer(model_file).unwrap();
                // measure time for prediction
                let start = Instant::now();
                let result = bst.predict_from_dmat(&xg_matrix, shp, &mut out_dim);
                let end = start.elapsed();
                println!("prediction time: {:?}", end);
                result.unwrap()
            })
            .flatten_iter()
            .collect();

        Grid2D::new(grid_shape, res, Some(OUT_NO_DATA_VALUE)).expect("raster creation must succeed")
    })
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

        for band in &self.sources {
            let stream = band.query(query, ctx).await.unwrap();
            band_buffer.push(stream);
        }

        let zipped_stream_tiled_bands = StreamVectorZip::new(band_buffer);

        let s: Vec<_> = zipped_stream_tiled_bands.collect().await;
        let strm = stream::iter(s).map(Ok).boxed();
        let rs =
            strm.and_then(move |tile| self.process_tile_async(tile, ctx.thread_pool().clone()));

        Ok(rs.boxed())
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::{
        MockExecutionContext, MockQueryContext, MultipleRasterSources, QueryProcessor,
        RasterOperator, RasterQueryProcessor, RasterResultDescriptor,
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
    use csv::Writer;
    use futures::StreamExt;
    use geoengine_datatypes::dataset::{DatasetId, InternalDatasetId};
    use geoengine_datatypes::primitives::{
        Coordinate2D, Measurement, QueryRectangle, RasterQueryRectangle, SpatialPartition2D,
        SpatialResolution, TimeGranularity, TimeInstance, TimeInterval, TimeStep,
    };
    use geoengine_datatypes::raster::{
        GridOrEmpty, RasterDataType, RasterPropertiesEntryType, RasterTile2D, TilingSpecification,
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

        for path in paths {
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
                        origin_coordinate: (474_112.0, 5_646_336.0).into(),
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
                start: TimeInstance::from_millis(1_072_917_000_000).unwrap(),
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
            (474_112.000, 5_646_336.000).into(),
            (522_752.000, 5_612_026.000).into(),
        )
        .unwrap();
        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = QueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1_590_969_600_000, 1_590_969_600_000).unwrap(),
            spatial_resolution: query_spatial_resolution,
        };

        let mut stream = rqp_gt.raster_query(qry_rectangle, &ctx).await.unwrap();

        let mut tile_buffer: Vec<_> = Vec::new();

        let props = test_util::create_properties(None, None, Some(11.0), Some(2.0));

        while let Some(processor) = stream.next().await {
            let mut tile = processor.unwrap();
            tile.properties = props.clone();
            tile_buffer.push(tile);
        }

        tile_buffer
    }

    /// Build a `MockRasterSource` to use with the operator
    async fn get_src(gcm_vec: Vec<GdalMetaDataRegular>) -> Vec<Box<dyn RasterOperator>> {
        let mut src_vec = Vec::new();

        for gcm in gcm_vec {
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
            (474_112.000, 5_646_336.000).into(),
            (522_752.000, 5_612_026.000).into(),
        )
        .unwrap();

        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = RasterQueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1_590_969_600_000, 1_590_969_600_000).unwrap(),
            spatial_resolution: query_spatial_resolution,
        };

        // this operator prepares the input data
        let srcs = get_src(gcm.clone()).await;

        let work_dir = std::env::current_dir().unwrap();
        let model_path = work_dir.join("model.json").display().to_string();

        // xg-operator takes the input data for further processing
        let xg = XgboostOperator {
            params: XgboostParams {
                model_file_path: model_path,
            },
            sources: MultipleRasterSources { rasters: srcs },
        };

        let op = RasterOperator::boxed(xg).initialize(&ctx).await.unwrap();

        let processor = op.query_processor().unwrap().get_f32().unwrap();

        let ctx = MockQueryContext::test_default();
        let result_stream = processor.query(qry_rectangle, &ctx).await.unwrap();
        let result: Vec<crate::util::Result<RasterTile2D<f32>>> = result_stream.collect().await;

        let mut all_pixels = Vec::new();

        for tile in result {
            let data_of_tile = tile.unwrap().into_materialized_tile().grid_array.data;
            for pixel in &data_of_tile {
                all_pixels.push(*pixel);
            }
        }

        let mut wtr = Writer::from_path("predictions.csv").unwrap();

        for elem in all_pixels {
            let num = format!("{elem}");
            wtr.write_record(&[&num]).unwrap();
        }
        wtr.flush().unwrap();
    }
}
