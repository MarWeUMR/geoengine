pub mod add_from_directory;
mod create_from_workflow;
pub mod external; // TODO: move to layers/external
pub mod in_memory;
pub mod listing;
pub mod storage;
pub mod upload;

#[cfg(feature = "xgboost")]
pub(crate) use create_from_workflow::{
    schedule_ml_model_from_workflow_task, MachineLearningModelFromWorkflow,
};
pub(crate) use create_from_workflow::{
    schedule_raster_dataset_from_workflow_task, RasterDatasetFromWorkflow,
    RasterDatasetFromWorkflowResult,
};

#[cfg(feature = "xgboost")]
#[cfg(test)] //TODO: remove test config, once its used outside of tests
pub(crate) use create_from_workflow::MachineLearningModelFromWorkflowResult;
