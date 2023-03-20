use actix_web::{web, Responder};
use futures::StreamExt;
use geoengine_datatypes::primitives::VectorQueryRectangle;

use geoengine_datatypes::raster::GridOrEmpty;
use geoengine_operators::call_on_generic_raster_processor;
use geoengine_operators::engine::{
    ExecutionContext, InitializedRasterOperator, QueryContext, QueryProcessor, RasterOperator,
    TypedRasterQueryProcessor,
};
use num_traits::AsPrimitive;

use super::tasks::TaskResponse;
use crate::contexts::Context;
use crate::error::Error;
use crate::error::Result;
use crate::machine_learning::{
    schedule_ml_model_from_workflow_task, Aggregatable, MLTrainRequest, MachineLearningAggregator,
    MachineLearningFeature, SimpleAggregator,
};
use crate::workflows::workflow::Workflow;

#[utoipa::path(
    tag = "Machine Learning",
    post,
    path = "/ml/train",
    request_body = MLTrainRequest,
    responses(
        (
            status = 200, description = "Model training from workflows", body = TaskResponse,
            example = json!({"taskId": "7f8a4cfe-76ab-4972-b347-b197e5ef0f3c"})
        )
    ),
    security(
        ("session_token" = [])
    )
)]
pub async fn ml_model_from_workflow_handler<C: Context>(
    session: C::Session,
    ctx: web::Data<C>,
    info: web::Json<MLTrainRequest>,
) -> Result<impl Responder> {
    let task_id = schedule_ml_model_from_workflow_task(
        info.input_workflows.clone(),
        info.label_workflow.clone(),
        session,
        ctx.into_inner(),
        info.into_inner(),
    )
    .await?;

    Ok(web::Json(TaskResponse::new(task_id)))
}

/// Build ML Features from the raw data and assign feature names.
pub async fn accumulate_raster_data(
    feature_names: Vec<Option<String>>,
    processors: Vec<TypedRasterQueryProcessor>,
    query: VectorQueryRectangle,
    qry_ctx: &dyn QueryContext,
    accum_variant: MachineLearningAggregator,
) -> Result<Vec<Result<MachineLearningFeature>>> {
    let collected_ml_features = processors
        .iter()
        .zip(feature_names.iter())
        .enumerate()
        .map(|(feature_pos, (op, feature_name))| {
            let name: String = if let Some(name) = feature_name {
                name.clone()
            } else {
                format!("feature_{feature_pos}")
            };

            let ml_feature = match accum_variant {
                MachineLearningAggregator::Simple => {
                    let aggregator = SimpleAggregator::<f32> { data: Vec::new() };
                    let feature = process_raster::<SimpleAggregator<f32>>(
                        name, op, query, qry_ctx, aggregator,
                    );
                    feature
                }
                MachineLearningAggregator::ReservoirSampling => {
                    todo!()
                }
            };

            ml_feature
        })
        .collect::<Vec<_>>();

    let results: Vec<Result<MachineLearningFeature>> =
        futures::future::join_all(collected_ml_features).await;
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

/// Collect data from a raster source into a ML Feature
async fn process_raster<C: Aggregatable<Data = Vec<f32>>>(
    name: String,
    input_rpq: &TypedRasterQueryProcessor,
    query: VectorQueryRectangle,
    ctx: &dyn QueryContext,
    mut aggregator_variant: C,
) -> Result<MachineLearningFeature> {
    call_on_generic_raster_processor!(input_rpq, processor => {

        let mut stream = processor.query(query.into(), ctx).await?;

        while let Some(tile) = stream.next().await {
            let tile = tile?;

            match tile.grid_array {
                // Ignore empty grids if no_data should not be included
                GridOrEmpty::Empty(_) => {},
                GridOrEmpty::Grid(grid) => {
                    // fetch the tile data
                    let values = grid
                        .masked_element_deref_iterator()
                        .filter_map(|pixel_option| {
                            pixel_option.map(|p| {
                                let v: f32 = p.as_();
                                v
                            })
                        }).collect(); // TODO: make this work for more types

                    // now let the accumulator collect the data
                    aggregator_variant.aggregate(values);
                }
            }
        }

        let collected_data = aggregator_variant.finish();

        Ok(MachineLearningFeature::new(Some(name), collected_data))

    })
}
