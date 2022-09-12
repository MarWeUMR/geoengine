use std::sync::Arc;

use async_trait::async_trait;
use futures::future;
use geoengine_datatypes::hashmap;
use geoengine_datatypes::primitives::{ClassificationMeasurement, Measurement};
use geoengine_datatypes::raster::{BaseTile, GridOrEmpty, GridShape, RasterDataType, Pixel, RasterTile2D};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};

use crate::engine::{
    ExecutionContext, InitializedRasterOperator, MultipleRasterSources, Operator, RasterOperator,
    RasterResultDescriptor, TypedRasterQueryProcessor, RasterQueryProcessor,
};
use crate::util::Result;
use futures::stream::BoxStream;
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
            data_type: RasterOut,
            time: None,
            bbox: None,
            spatial_reference: in_desc.spatial_reference,
            measurement: Measurement::Classification(ClassificationMeasurement {
                measurement: "raw".into(),
                classes: hashmap!(0 => "Water Bodies".to_string()),
            }),
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

        // extract the actual data from the tiles
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

        let model = self.model_file.clone();
        let predicted_grid = crate::util::spawn_blocking(move || {
            process_tile(
                &pixels,
                &pool,
                model.as_bytes(),
                grid_shape,
                n_bands as usize,
            )
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
