use serde::{Deserialize, Serialize};

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
