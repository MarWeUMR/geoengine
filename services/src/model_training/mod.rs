mod ml_tasks;
pub mod xg_error;
mod xgboost_training;

pub(crate) use ml_tasks::{schedule_ml_model_from_workflow_task, MLTrainRequest};

#[cfg(test)] //TODO: remove test config, once its used outside of tests
pub(crate) use ml_tasks::MachineLearningModelFromWorkflowResult;

pub(crate) use xgboost_training::xgb_train_model;
pub(crate) use xgboost_training::XgboostTrainingParams;
