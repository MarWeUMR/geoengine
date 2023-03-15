use crate::contexts::Context;
use crate::datasets::upload::UploadId;
use crate::error;
use crate::handlers::model_training::{
    accumulate_raster_data, get_operators_from_workflows, get_query_processors,
};
use crate::tasks::{Task, TaskId, TaskManager, TaskStatusInfo};
use crate::workflows::workflow::Workflow;
use geoengine_datatypes::error::ErrorSource;
use geoengine_datatypes::primitives::MachineLearningQueryRectangle;
use geoengine_datatypes::util::Identifier;
use geoengine_operators::engine::ExecutionContext;
use serde::{Deserialize, Serialize};
use snafu::ResultExt;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use utoipa::ToSchema;

use super::{xgb_train_model, XgboostTrainingParams};

use serde_json::Value;

/// By default, we set [`RasterDatasetFromWorkflow::as_cog`] to true to produce cloud-optmized `GeoTiff`s.
#[inline]
const fn default_as_cog() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(example = json!({"name": "foo", "description": null, "query": {"spatialBounds": {"upperLeftCoordinate": {"x": -10.0, "y": 80.0}, "lowerRightCoordinate": {"x": 50.0, "y": 20.0}}, "timeInterval": {"start": 1_388_534_400_000_i64, "end": 1_388_534_401_000_i64}, "spatialResolution": {"x": 0.1, "y": 0.1}}}))]
pub struct MLTrainRequest {
    pub params: XgboostTrainingParams,
    pub input_workflows: Vec<Workflow>,
    pub label_workflow: Vec<Workflow>,
    pub query: MachineLearningQueryRectangle,
}

/// parameter for the dataset from workflow handler (body)
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
#[schema(example = json!({"name": "foo", "description": null, "query": {"spatialBounds": {"upperLeftCoordinate": {"x": -10.0, "y": 80.0}, "lowerRightCoordinate": {"x": 50.0, "y": 20.0}}, "timeInterval": {"start": 1_388_534_400_000_i64, "end": 1_388_534_401_000_i64}, "spatialResolution": {"x": 0.1, "y": 0.1}}}))]
pub struct MachineLearningModelFromWorkflow {
    pub name: String,
    pub description: Option<String>,
    pub query: MachineLearningQueryRectangle,

    // TODO: is that cog stuff relevant in this case?
    #[schema(default = default_as_cog)]
    #[serde(default = "default_as_cog")]
    pub as_cog: bool,
}

/// response of the machine learning model from workflow handler
#[cfg(feature = "xgboost")]
#[derive(Clone, Debug, Deserialize, Serialize, ToSchema)]
pub struct MachineLearningModelFromWorkflowResult {
    pub store_path: PathBuf,
    pub model: Value,
}

#[cfg(feature = "xgboost")]
impl TaskStatusInfo for MachineLearningModelFromWorkflowResult {}

#[cfg(feature = "xgboost")]
pub struct MachineLearningModelFromWorkflowTask<C: Context> {
    pub input_workflows: Vec<Workflow>,
    pub label_workflow: Vec<Workflow>,
    pub session: C::Session,
    pub ctx: Arc<C>,
    pub info: MLTrainRequest,
    pub query: MachineLearningQueryRectangle,
    pub upload: UploadId, // TODO: remove?
    pub store_path: PathBuf,
}

#[cfg(feature = "xgboost")]
impl<C: Context> MachineLearningModelFromWorkflowTask<C> {
    async fn process(&self) -> error::Result<MachineLearningModelFromWorkflowResult> {
        let train_cfg = self.info.params.training_config.clone();
        let store_path = self.store_path.clone();
        let feature_names = self.info.params.feature_names.clone();

        // put the labels workflow at the end of the workflow vec
        let mut inputs = self.info.input_workflows.clone();
        let labels = self.info.label_workflow.clone();
        inputs.extend(labels);

        let input_operators = get_operators_from_workflows(inputs)?;

        let mut exe_ctx = self.ctx.execution_context(self.session.clone())?;

        let typed_query_processors = get_query_processors(input_operators, &exe_ctx).await?;

        let query = self.query;
        let query_ctx = self.ctx.query_context(self.session.clone())?;

        let mut accumulated_data =
            accumulate_raster_data(feature_names, typed_query_processors, query, &query_ctx)
                .await?;

        //TODO: this could be wrapped by a match or even a dedicated ml model struct,
        // which unifies the different kind of models that could be generated
        let model = xgb_train_model(&mut accumulated_data, train_cfg)?;

        exe_ctx
            .write_ml_model(store_path.clone(), model.to_string())
            .await?;

        Ok(MachineLearningModelFromWorkflowResult { store_path, model })
    }
}

#[cfg(feature = "xgboost")]
#[async_trait::async_trait]
impl<C: Context> Task<C::TaskContext> for MachineLearningModelFromWorkflowTask<C> {
    async fn run(
        &self,
        _ctx: C::TaskContext,
    ) -> Result<Box<dyn crate::tasks::TaskStatusInfo>, Box<dyn ErrorSource>> {
        let response = self.process().await;

        response
            .map(TaskStatusInfo::boxed)
            .map_err(ErrorSource::boxed)
    }

    async fn cleanup_on_error(&self, _ctx: C::TaskContext) -> Result<(), Box<dyn ErrorSource>> {
        fs::remove_dir_all(&self.store_path)
            .await
            .context(crate::error::Io)
            .map_err(ErrorSource::boxed)?;

        //TODO: Dataset might already be in the database, if task was already close to finishing.

        Ok(())
    }

    fn task_type(&self) -> &'static str {
        "create-ml-model"
    }

    fn task_unique_id(&self) -> Option<String> {
        Some(self.upload.to_string())
    }
}

#[cfg(feature = "xgboost")]
pub async fn schedule_ml_model_from_workflow_task<C: Context>(
    input_workflows: Vec<Workflow>,
    label_workflow: Vec<Workflow>,
    session: C::Session,
    ctx: Arc<C>,
    info: MLTrainRequest,
) -> error::Result<TaskId> {
    let store_path = info
        .params
        .model_store_path
        .clone()
        .ok_or_else(|| super::xg_error::XGBoostModuleError::CouldNotGetMlModelPath)?;
    let upload = UploadId::new();

    let task = MachineLearningModelFromWorkflowTask {
        input_workflows,
        label_workflow,
        session,
        ctx: ctx.clone(),
        info: info.clone(),
        upload,
        query: info.query,
        store_path,
    }
    .boxed();

    let task_id = ctx.tasks_ref().schedule(task, None).await?;

    Ok(task_id)
}
