use std::sync::Arc;

use actix_web::{web, Responder};
use futures::stream::select_all;
use futures::{FutureExt, StreamExt, TryFutureExt, TryStreamExt};
use geoengine_datatypes::primitives::VectorQueryRectangle;

use geoengine_datatypes::raster::{ConvertDataTypeParallel, GridOrEmpty};
use geoengine_operators::call_on_generic_raster_processor;
use geoengine_operators::engine::{
    ExecutionContext, InitializedRasterOperator, QueryContext, QueryProcessor, RasterOperator,
    TypedRasterQueryProcessor,
};
use geoengine_operators::util::spawn_blocking_with_thread_pool;

use super::tasks::TaskResponse;
use crate::contexts::ApplicationContext;
use crate::error::Error;
use crate::error::Result;
use crate::machine_learning::{
    schedule_ml_model_from_workflow_task, xg_error::XGBoostModuleError as XgModuleError,
    Aggregatable, MLTrainRequest, MachineLearningAggregator, MachineLearningFeature,
    ReservoirSamplingAggregator, SimpleAggregator,
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
pub async fn ml_model_from_workflow_handler<C: ApplicationContext>(
    session: C::Session,
    app_ctx: web::Data<C>,
    info: web::Json<MLTrainRequest>,
) -> Result<impl Responder> {
    let ctx = Arc::new(app_ctx.session_context(session));

    let task_id = schedule_ml_model_from_workflow_task(
        info.input_workflows.clone(),
        info.label_workflow.clone(),
        ctx,
        info.into_inner(),
    )
    .await?;

    Ok(web::Json(TaskResponse::new(task_id)))
}

/// This method initializes the raster operators and produces a vector of typed raster query
/// processors.
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

/// This method forwards the collected tile data to the appropriate implementation of
/// the actual aggregation algorithm.
fn accumulate_tile_data(
    tile_data: Vec<Option<f32>>,
    accumulator: &mut Box<dyn Aggregatable<Data = Vec<f32>> + Send>,
) {
    let v = tile_data.into_iter().flatten().collect::<Vec<f32>>();

    accumulator.aggregate(v);
}

//TODO: add a way to abort the query execution when the tasks is aborted
/// Build ML Features from the raw data and assign feature names.
pub async fn accumulate_raster_data(
    feature_names: Vec<Option<String>>,
    processors: Vec<TypedRasterQueryProcessor>,
    query: VectorQueryRectangle,
    ctx: &dyn QueryContext,
    accum_variant: MachineLearningAggregator,
) -> Result<Vec<MachineLearningFeature>> {
    type MlData = Box<dyn Aggregatable<Data = Vec<f32>> + Send>;

    let mut queries = Vec::with_capacity(processors.len());
    let q = query.into();
    for (i, raster_processor) in processors.iter().enumerate() {
        queries.push(
            call_on_generic_raster_processor!(raster_processor, processor => {
                processor.query(q, ctx).await?
                         .and_then(
                            move |tile| spawn_blocking_with_thread_pool(
                                ctx.thread_pool().clone(),
                                move || (i, tile.convert_data_type_parallel()) ).map_err(Into::into)
                        ).boxed()
            }),
        );
    }

    let mut aggregator_vec = Vec::new();
    let n_rasters = processors.len();

    // populate the aggregator vector with an aggregator for each raster
    match accum_variant {
        MachineLearningAggregator::Simple => {
            (0..n_rasters).into_iter().for_each(|_| {
                let aggregator = SimpleAggregator::<f32> { data: Vec::new() };
                aggregator_vec.push(Box::new(aggregator) as MlData);
            });
        }
        MachineLearningAggregator::ReservoirSampling => {
            (0..n_rasters).into_iter().for_each(|_| {
                let aggregator = ReservoirSamplingAggregator::<f32> { data: Vec::new() };
                aggregator_vec.push(Box::new(aggregator) as MlData);
            });
        }
    }

    let result = select_all(queries)
        .fold(
            Ok(aggregator_vec),
            |aggregator_vec: Result<Vec<MlData>>, enumerated_raster_tile| async move {
                let mut aggregator_vec = aggregator_vec?;
                let (i, raster_tile) = enumerated_raster_tile?;

                let res = match raster_tile.grid_array {
                    GridOrEmpty::Grid(g) => {
                        let agg = aggregator_vec
                            .get_mut(i)
                            .ok_or(XgModuleError::CouldNotGetMlAggregatorRef { index: i })?;

                        let tile_data: Vec<_> = g.masked_element_deref_iterator().collect();

                        accumulate_tile_data(tile_data, agg);

                        Ok(aggregator_vec)
                    }
                    GridOrEmpty::Empty(_) => Ok(aggregator_vec),
                };

                res
            },
        )
        .map(|aggregator_vec| {
            aggregator_vec?
                .into_iter()
                .enumerate()
                .map(|(i, agg)| {
                    let name = feature_names
                        .get(i)
                        .ok_or(XgModuleError::CouldNotGetMlFeatureName { index: i })?;
                    let collected_data = agg.finish();
                    let ml_feature =
                        MachineLearningFeature::new(name.clone(), collected_data.clone());

                    Ok(ml_feature)
                })
                .collect::<Result<Vec<MachineLearningFeature>>>()
        })
        .await;

    result
}
