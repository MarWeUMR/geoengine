use futures::StreamExt;
use geoengine_datatypes::primitives::MachineLearningQueryRectangle;

#[cfg(feature = "xgboost")]
use geoengine_datatypes::pro::MachineLearningFeature;
use geoengine_datatypes::raster::GridOrEmpty;
use geoengine_operators::call_on_generic_raster_processor;
use geoengine_operators::engine::{
    ExecutionContext, InitializedRasterOperator, QueryContext, QueryProcessor, RasterOperator,
    TypedRasterQueryProcessor,
};
use num_traits::AsPrimitive;

use crate::error::Error;
use crate::error::Result;
use crate::workflows::workflow::Workflow;

const BATCH_SIZE: usize = 1_000;

#[derive(Debug)]
enum MachineLearningAccumKind {
    XGBoost(Vec<f32>),
}

#[cfg(feature = "xgboost")]
#[derive(Debug)]
/// Used to gather the source data before further processing/shaping for xgboost
struct MachineLearningAccum {
    feature_name: String,
    accum: MachineLearningAccumKind,
}

impl MachineLearningAccum {
    fn new(feature_name: String) -> MachineLearningAccum {
        MachineLearningAccum {
            feature_name,
            accum: MachineLearningAccumKind::XGBoost(Vec::new()),
        }
    }

    fn update(&mut self, values: impl Iterator<Item = f32>) {
        match self.accum {
            MachineLearningAccumKind::XGBoost(ref mut accumulator) => {
                for chunk in &itertools::Itertools::chunks(values, BATCH_SIZE) {
                    accumulator.extend(chunk.filter(|x| x.is_finite()));
                }
            }
        }
    }

    fn finish(&mut self) -> MachineLearningFeature {
        match &self.accum {
            MachineLearningAccumKind::XGBoost(data) => {
                MachineLearningFeature::new(Some(self.feature_name.clone()), data.clone())
            }
        }
    }
}

/// Build ML Features from the raw data and assign feature names.
pub async fn accumulate_raster_data(
    feature_names: Vec<Option<String>>,
    input: Vec<TypedRasterQueryProcessor>,
    query: MachineLearningQueryRectangle,
    qry_ctx: &dyn QueryContext,
) -> Result<Vec<Result<MachineLearningFeature>>> {
    let mut feature_counter = -1;

    let data = input
        .iter()
        .zip(feature_names.iter())
        .map(|(op, feature_name)| {
            let name: String = if let Some(name) = feature_name {
                name.clone()
            } else {
                feature_counter += 1;
                format!("feature_{feature_counter}")
            };
            process_raster(name, op, query, qry_ctx)
        })
        .collect::<Vec<_>>();

    let results: Vec<Result<MachineLearningFeature>> = futures::future::join_all(data).await;
    Ok(results)
}

pub async fn get_query_processors(
    operators: Vec<Box<dyn RasterOperator>>,
    exe_ctx: &dyn ExecutionContext,
) -> Result<Vec<TypedRasterQueryProcessor>> {
    let initialized = futures::future::join_all(operators.into_iter().map(|op| {
        let init = op.initialize(exe_ctx);
        init
    }))
    .await
    .into_iter()
    .collect::<Vec<_>>();

    let query_processors = initialized
        .into_iter()
        .map(|init_raster_op| {
            let query_processor = init_raster_op?.query_processor()?;
            Ok(query_processor)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(query_processors)
}

pub fn get_operators_from_workflows(
    workflows: Vec<Workflow>,
) -> Result<Vec<Box<dyn RasterOperator>>> {
    let initialized_raster_operators = workflows
        .into_iter()
        .map(|workflow| {
            let op = workflow.operator;
            let raster_op = op.get_raster()?;
            Ok(raster_op)
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(initialized_raster_operators)
}

/// Collect data from a raster source to an ML Accumulator
async fn process_raster(
    name: String,
    input_rpq: &TypedRasterQueryProcessor,
    query: MachineLearningQueryRectangle,
    ctx: &dyn QueryContext,
) -> Result<MachineLearningFeature> {
    call_on_generic_raster_processor!(input_rpq, processor => {

        let mut stream = processor.query(query.into(), ctx).await?;
        let mut accum = MachineLearningAccum::new(name);


        while let Some(tile) = stream.next().await {
            let tile = tile?;

            match tile.grid_array {
                // Ignore empty grids if no_data should not be included
                GridOrEmpty::Empty(_) => {},
                GridOrEmpty::Grid(grid) => {
                    accum.update(
                        grid.masked_element_deref_iterator().filter_map(|pixel_option| {
                            pixel_option.map(|p| { let v: f32 = p.as_(); v})
                        }));
                }
            }
        }

        Ok(accum.finish())
    })
}
