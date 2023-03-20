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

/// This enum represents the different aggregators that can be used to initialize the
/// different algorithms for collecting the data used in ml training.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum MachineLearningAggregator {
    Simple,
    ReservoirSampling,
}

/// The purpose of this trait is to facilitate a generic aggregation process of raster data.
pub trait Aggregatable {
    type Data;

    /// This method should realize the aggregation algorithm of the implementing struct.
    fn aggregate(&mut self, incoming_data: Self::Data);

    /// Once the aggregation process is finished, return the data for further usage.
    fn finish(self) -> Self::Data;
}

/// A simple aggregator that just collects all incoming data in a vector.
pub struct SimpleAggregator<T> {
    pub data: Vec<T>,
}

impl<T> Aggregatable for SimpleAggregator<T> {
    type Data = Vec<T>;

    fn aggregate(&mut self, incoming_data: Vec<T>) {
        self.data.extend(incoming_data);
    }

    fn finish(self) -> Vec<T> {
        self.data
    }
}

/// A reservoir sampling aggregator that samples a given number of elements from the incoming data.
/// This aggregator can be used, when the data size exceeds the available memory.
pub struct ReservoirSamplingAggregator<T> {
    pub data: Vec<T>,
}

// TODO: implement reservoir sampling
impl<T> Aggregatable for ReservoirSamplingAggregator<T> {
    type Data = Vec<T>;

    fn aggregate(&mut self, incoming_data: Vec<T>) {
        self.data.extend(incoming_data);
    }

    fn finish(self) -> Vec<T> {
        self.data
    }
}
