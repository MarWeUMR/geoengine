use serde::{Deserialize, Serialize};

// This is an inclusion point of Geo Engine Pro

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Serialize)]
pub enum MachineLearningModelOutputFormat {
    JsonPlain,
    // fancy tensorflow model output format
    // LightGBM
    // etc...
}
