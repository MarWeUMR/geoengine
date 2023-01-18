use serde::{Deserialize, Serialize};

// This is an inclusion point of Geo Engine Pro
mod xgboost;

pub use xgboost::MachineLearningFeature;

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Serialize)]
pub enum MachineLearningModelOutputFormat {
    JsonPlain,
    // fancy tensorflow model output format
    // LightGBM
    // etc...
}
