use std::collections::hash_map::Entry;
use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};
use typetag::serde;

use serde_json::Value;
use snafu::{ensure, ResultExt};
use xgboost_rs::{Booster, DMatrix};

use crate::error::Result;
use crate::model_training::xg_error as XgModuleError;
use geoengine_datatypes::pro::MachineLearningFeature;
use ordered_float::NotNan;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct XgboostTrainingParams {
    // where to store the model file
    pub model_store_path: Option<PathBuf>,
    pub no_data_value: f32, // FIXME: remove?
    pub training_config: HashMap<String, String>,
    pub feature_names: Vec<Option<String>>,
}

pub fn xgb_train_model(
    results: &mut Vec<Result<MachineLearningFeature>>,
    training_config: HashMap<String, String>,
) -> Result<Value> {
    ensure!(
        results.len() >= 2,
        XgModuleError::error::MachineLearningMustHaveAtLeastTwoFeatures
    );

    let n_bands = results.len() - 1;
    let lbls = results
        .pop()
        .expect("There should have been at least two features!");

    let raw_data = results
        .iter_mut()
        .map(|elem| {
            let feature_data: &Vec<f32> = &elem
                .as_ref()
                .map_err(|_| {
                    XgModuleError::XGBoostModuleError::MachineLearningFeatureDataNotAvailable
                })?
                .feature_data;
            Ok(feature_data)
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    let n_rows = raw_data.len() / n_bands;

    let strides_ax_0 = 1;
    let strides_ax_1 = n_rows;
    let byte_size_ax_0 = std::mem::size_of::<f32>() * strides_ax_0;
    let byte_size_ax_1 = std::mem::size_of::<f32>() * strides_ax_1;

    let mut dmatrix = DMatrix::from_col_major_f32(
        raw_data
            .into_iter()
            .copied()
            .collect::<Vec<f32>>()
            .as_slice(),
        byte_size_ax_0,
        byte_size_ax_1,
        n_rows,
        n_bands,
        -1,
        f32::NAN,
    )
    .context(XgModuleError::error::CreateDMatrix)?;

    let lbls_remap = remap_labels(lbls?.feature_data)?;

    dmatrix
        .set_labels(lbls_remap.as_slice())
        .context(XgModuleError::error::DMatrixSetLabels)?; // <- here we need the remapped target values

    let evals = &[(&dmatrix, "train")];
    let bst = Booster::train(
        Some(evals),
        &dmatrix,
        training_config,
        None, // <- No old model yet
    )
    .context(XgModuleError::error::BoosterTraining)?;

    let model = bst
        .save_to_buffer("json".into())
        .context(XgModuleError::error::ModelStorage)?;

    let res = serde_json::from_str(model.as_str())?;
    Ok(res)
}

fn remap_labels(lbls: Vec<f32>) -> Result<Vec<f32>> {
    let mut unique_lbl_counter: f32 = -1.0;

    // TODO: persist this hashmap?
    let mut unique_lbls_hm: HashMap<NotNan<f32>, f32> = HashMap::new();

    let remapped_values = lbls
        .into_iter()
        .map(|elem| {
            let key: NotNan<f32> = NotNan::new(elem)?;

            if let Entry::Vacant(e) = unique_lbls_hm.entry(key) {
                unique_lbl_counter += 1.0;
                e.insert(unique_lbl_counter);
                Ok(unique_lbl_counter)
            } else {
                // return the already remapped value, instead of the original
                let remapped_val = unique_lbls_hm
                    .get(&key)
                    .ok_or_else(|| XgModuleError::XGBoostModuleError::CouldNotGetMlLabelKeyName)?;
                Ok(*remapped_val)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(remapped_values)
}
