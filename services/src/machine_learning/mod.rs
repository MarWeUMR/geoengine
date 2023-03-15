mod ml_tasks;
pub mod xg_error;
mod xgboost_training;

pub(crate) use ml_tasks::{schedule_ml_model_from_workflow_task, MLTrainRequest};

#[cfg(test)] //TODO: remove test config, once its used outside of tests
pub(crate) use ml_tasks::MachineLearningModelFromWorkflowResult;

use serde::Deserialize;
use serde::Serialize;
use typetag::serde;
pub(crate) use xgboost_training::xgb_train_model;
pub(crate) use xgboost_training::XgboostTrainingParams;

/// Represents a single box of a box plot including whiskers
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineLearningFeature {
    pub feature_name: Option<String>, // e.g. temperature, elevation etc.
    pub feature_data: Vec<f32>,
}

impl MachineLearningFeature {
    pub fn new(feature_name: Option<String>, feature_data: Vec<f32>) -> MachineLearningFeature {
        MachineLearningFeature {
            feature_name,
            feature_data,
        }
    }
}
