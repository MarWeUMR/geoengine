use indexmap::IndexMap;
use libc;
use log::debug;
use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::{ffi, fmt, fs::File, ptr, slice};
use tempfile;
use xgboost_sys;

use crate::pro::ml::bindings::error::XGBResult;
use crate::pro::ml::bindings::parameters::{BoosterParameters, TrainingParameters};
use crate::pro::ml::bindings::XGBError;

use super::DMatrix;

pub type CustomObjective = fn(&[f32], &DMatrix) -> (Vec<f32>, Vec<f32>);

/// Used to control the return type of predictions made by C Booster API.
enum PredictOption {
    OutputMargin,
    PredictLeaf,
    PredictContribitions,
    //ApproximateContributions,
    PredictInteractions,
}

impl PredictOption {
    /// Convert list of options into a bit mask.
    fn options_as_mask(options: &[PredictOption]) -> i32 {
        let mut option_mask = 0x00;
        for option in options {
            let value = match *option {
                PredictOption::OutputMargin => 0x01,
                PredictOption::PredictLeaf => 0x02,
                PredictOption::PredictContribitions => 0x04,
                //PredictOption::ApproximateContributions => 0x08,
                PredictOption::PredictInteractions => 0x10,
            };
            option_mask |= value;
        }

        option_mask
    }
}

/// Core model in `XGBoost`, containing functions for training, evaluating and predicting.
///
/// Usually created through the [`train`](struct.Booster.html#method.train) function, which
/// creates and trains a Booster in a single call.
///
/// For more fine grained usage, can be created using [`new`](struct.Booster.html#method.new) or
/// [`new_with_cached_dmats`](struct.Booster.html#method.new_with_cached_dmats), then trained by calling
/// [`update`](struct.Booster.html#method.update) or [`update_custom`](struct.Booster.html#method.update_custom)
/// in a loop.
#[derive(Clone)]
pub struct Booster {
    handle: xgboost_sys::BoosterHandle,
}

unsafe impl Send for Booster {}
unsafe impl Sync for Booster {}

impl Booster {
    /// Create a new Booster model with given parameters.
    ///
    /// This model can then be trained using calls to update/boost as appropriate.
    ///
    /// The [`train`](struct.Booster.html#method.train)  function is often a more convenient way of constructing,
    /// training and evaluating a Booster in a single call.
    pub fn new(params: &BoosterParameters) -> XGBResult<Self> {
        Self::new_with_cached_dmats(params, &[])
    }

    pub fn new_with_json_config(
        dmats: &[&DMatrix],

        config: HashMap<&str, &str>,
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe if any dmats are freed
        let s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_sys::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        booster.set_param_from_json(config);
        Ok(booster)
    }

    /// Create a new booster model with given parameters and list of `DMatrix` to cache.
    ///
    /// Cached `DMatrix` can sometimes be used internally by `XGBoost` to speed up certain operations.
    pub fn new_with_cached_dmats(
        params: &BoosterParameters,
        dmats: &[&DMatrix],
    ) -> XGBResult<Self> {
        let mut handle = ptr::null_mut();
        // TODO: check this is safe if any dmats are freed
        let s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();
        xgb_call!(xgboost_sys::XGBoosterCreate(
            s.as_ptr(),
            dmats.len() as u64,
            &mut handle
        ))?;

        let mut booster = Booster { handle };
        booster.set_params(params)?;
        Ok(booster)
    }

    /// Save this Booster as a binary file at given path.
    ///
    /// # Panics
    ///
    /// Will panic, if the model saving fails with an error not coming from `XGBoost`.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> XGBResult<()> {
        debug!("Writing Booster to: {}", path.as_ref().display());
        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSaveModel(self.handle, fname.as_ptr()))
    }

    /// Load a `Booster` from a binary file at given path.
    ///
    /// # Panics
    ///
    /// Will panic, if the model couldn't be loaded, because of an error not coming from `XGBoost`.
    /// Could also panic, if a `Booster` couldn't be created because of an error not coming from `XGBoost`.
    pub fn load<P: AsRef<Path>>(path: P) -> XGBResult<Self> {
        debug!("Loading Booster from: {}", path.as_ref().display());

        // gives more control over error messages, avoids stack trace dump from C++
        if !path.as_ref().exists() {
            return Err(XGBError::new(format!(
                "File not found: {}",
                path.as_ref().display()
            )));
        }

        let fname = ffi::CString::new(path.as_ref().as_os_str().as_bytes()).unwrap();
        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterLoadModel(handle, fname.as_ptr()))?;
        Ok(Booster { handle })
    }

    /// Load a Booster directly from a buffer.
    pub fn load_buffer(bytes: &[u8]) -> XGBResult<Self> {
        debug!("Loading Booster from buffer (length = {})", bytes.len());

        let mut handle = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterCreate(ptr::null(), 0, &mut handle))?;
        xgb_call!(xgboost_sys::XGBoosterLoadModelFromBuffer(
            handle,
            bytes.as_ptr().cast(),
            bytes.len() as u64
        ))?;
        Ok(Booster { handle })
    }

    /// Trains the model incrementally.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` fails to load the number of processes from `Rabit`.
    pub fn train_increment(params: &TrainingParameters, model_name: &str) -> XGBResult<Self> {
        let mut dmats = vec![params.dtrain];
        if let Some(eval_sets) = params.evaluation_sets {
            for (dmat, _) in eval_sets {
                dmats.push(*dmat);
            }
        }

        let path = Path::new(model_name);
        let bytes = std::fs::read(path).expect("can't read saved booster file");
        let mut bst = Booster::load_buffer(&bytes[..]).expect("can't load booster from buffer");

        // load distributed code checkpoint from rabit
        let version = bst.load_rabit_checkpoint()?;
        debug!("Loaded Rabit checkpoint: version={}", version);
        assert!(unsafe { xgboost_sys::RabitGetWorldSize() != 1 || version == 0 });

        unsafe { xgboost_sys::RabitGetRank() };
        let start_iteration = version / 2;

        for i in start_iteration..params.boost_rounds as i32 {
            // distributed code: need to resume to this point
            // skip first update if a recovery step
            if version % 2 == 0 {
                if let Some(objective_fn) = params.custom_objective_fn {
                    debug!("Boosting in round: {}", i);
                    bst.update_custom(params.dtrain, objective_fn)?;
                } else {
                    debug!("Updating in round: {}", i);
                    bst.update(params.dtrain, i)?;
                }
                bst.save_rabit_checkpoint()?;
            }

            assert!(unsafe {
                xgboost_sys::RabitGetWorldSize() == 1
                    || version == xgboost_sys::RabitVersionNumber()
            });

            //nboost += 1;

            if let Some(eval_sets) = params.evaluation_sets {
                let mut dmat_eval_results = bst.eval_set(eval_sets, i)?;

                if let Some(eval_fn) = params.custom_evaluation_fn {
                    let eval_name = "custom";
                    for (dmat, dmat_name) in eval_sets {
                        let margin = bst.predict_margin(dmat)?;
                        let eval_result = eval_fn(&margin, dmat);
                        let eval_results = dmat_eval_results
                            .entry(eval_name.to_string())
                            .or_insert_with(IndexMap::new);
                        eval_results.insert(String::from(*dmat_name), eval_result);
                    }
                }

                // convert to map of eval_name -> (dmat_name -> score)
                let mut eval_dmat_results = BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }

                print!("[{}]", i);
                for (eval_name, dmat_results) in eval_dmat_results {
                    for (dmat_name, result) in dmat_results {
                        print!("\t{}-{}:{}", dmat_name, eval_name, result);
                    }
                }
                println!();
            }
        }

        Ok(bst)
    }

    pub fn train(
        evaluation_sets: Option<&[(&DMatrix, &str)]>,
        dtrain: &DMatrix,
        config: HashMap<&str, &str>,
        bst: Option<Booster>,
    ) -> XGBResult<Self> {
        let cached_dmats = {
            let mut dmats = vec![dtrain];
            if let Some(eval_sets) = evaluation_sets {
                for (dmat, _) in eval_sets {
                    dmats.push(*dmat);
                }
            }
            dmats
        };

        let mut bst: Booster = {
            if let Some(booster) = bst {
                let mut length: u64 = 0;
                let mut buffer_string = ptr::null();

                xgb_call!(xgboost_sys::XGBoosterSerializeToBuffer(
                    booster.handle,
                    &mut length,
                    &mut buffer_string
                )).expect("couldn't serialize to buffer!");

                let mut bst_handle = ptr::null_mut();

                let cached_dmat_handles: Vec<xgboost_sys::DMatrixHandle> =
                    cached_dmats.iter().map(|x| x.handle).collect();

                xgb_call!(xgboost_sys::XGBoosterCreate(
                    cached_dmat_handles.as_ptr(),
                    cached_dmats.len() as u64,
                    &mut bst_handle
                ))?;

                let mut bst_unserialize = Booster { handle: bst_handle };

                xgb_call!(xgboost_sys::XGBoosterUnserializeFromBuffer(
                    bst_unserialize.handle,
                    buffer_string as *mut ffi::c_void,
                    length,
                )).expect("couldn't unserialize from buffer!");

                bst_unserialize.set_param_from_json(config);
                bst_unserialize
            } else {
                Booster::new_with_json_config(&cached_dmats, config)?
            }
        };

        for i in 0..16 {
            bst.update(dtrain, i)?;

            if let Some(eval_sets) = evaluation_sets {
                let dmat_eval_results = bst.eval_set(eval_sets, i)?;

                // convert to map of eval_name -> (dmat_name -> score)
                let mut eval_dmat_results = BTreeMap::new();
                for (dmat_name, eval_results) in &dmat_eval_results {
                    for (eval_name, result) in eval_results {
                        let dmat_results = eval_dmat_results
                            .entry(eval_name)
                            .or_insert_with(BTreeMap::new);
                        dmat_results.insert(dmat_name, result);
                    }
                }

                print!("[{}]", i);
                for (eval_name, dmat_results) in eval_dmat_results {
                    for (dmat_name, result) in dmat_results {
                        print!("\t{}-{}:{}", dmat_name, eval_name, result);
                    }
                }
                println!();
            }
        }

        Ok(bst)
    }

    /// Saves the config as a json file.
    ///
    /// # Panics
    ///
    /// Will panic, if the config cant be created, because of an error not coming from `XGBoost`.
    pub fn save_config(&self) -> String {
        let mut length: u64 = 1;
        let mut json_string = ptr::null();

        let json = unsafe {
            xgboost_sys::XGBoosterSaveJsonConfig(self.handle, &mut length, &mut json_string)
        };

        let out = unsafe {
            ffi::CStr::from_ptr(json_string)
                .to_str()
                .unwrap()
                .to_owned()
        };

        println!("{}", json);
        println!("{}", out);
        out
    }

    /// Update this Booster's parameters.
    pub fn set_params(&mut self, p: &BoosterParameters) -> XGBResult<()> {
        for (key, value) in p.as_string_pairs() {
            println!("challis: Setting parameter: {}={}", &key, &value);
            self.set_param(&key, &value)?;
        }
        Ok(())
    }

    /// Update this model by training it for one round with given training matrix.
    ///
    /// Uses `XGBoost`'s objective function that was specificed in this Booster's learning objective parameters.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `iteration` - current iteration number
    pub fn update(&mut self, dtrain: &DMatrix, iteration: i32) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterUpdateOneIter(
            self.handle,
            iteration,
            dtrain.handle
        ))
    }

    /// Update this model by training it for one round with a custom objective function.
    pub fn update_custom(
        &mut self,
        dtrain: &DMatrix,
        objective_fn: CustomObjective,
    ) -> XGBResult<()> {
        let pred = self.predict(dtrain)?;
        let (gradient, hessian) = objective_fn(&pred, dtrain);
        self.boost(dtrain, &gradient, &hessian)
    }

    /// Update this model by directly specifying the first and second order gradients.
    ///
    /// This is typically used instead of `update` when using a customised loss function.
    ///
    /// * `dtrain` - matrix to train the model with for a single iteration
    /// * `gradient` - first order gradient
    /// * `hessian` - second order gradient
    fn boost(&mut self, dtrain: &DMatrix, gradient: &[f32], hessian: &[f32]) -> XGBResult<()> {
        if gradient.len() != hessian.len() {
            let msg = format!(
                "Mismatch between length of gradient and hessian arrays ({} != {})",
                gradient.len(),
                hessian.len()
            );
            return Err(XGBError::new(msg));
        }
        assert_eq!(gradient.len(), hessian.len());

        // TODO: _validate_feature_names
        let mut grad_vec = gradient.to_vec();
        let mut hess_vec = hessian.to_vec();
        xgb_call!(xgboost_sys::XGBoosterBoostOneIter(
            self.handle,
            dtrain.handle,
            grad_vec.as_mut_ptr(),
            hess_vec.as_mut_ptr(),
            grad_vec.len() as u64
        ))
    }

    fn eval_set(
        &self,
        evals: &[(&DMatrix, &str)],
        iteration: i32,
    ) -> XGBResult<IndexMap<String, IndexMap<String, f32>>> {
        let (dmats, names) = {
            let mut dmats = Vec::with_capacity(evals.len());
            let mut names = Vec::with_capacity(evals.len());
            for (dmat, name) in evals {
                dmats.push(dmat);
                names.push(*name);
            }
            (dmats, names)
        };
        assert_eq!(dmats.len(), names.len());

        let mut s: Vec<xgboost_sys::DMatrixHandle> = dmats.iter().map(|x| x.handle).collect();

        // build separate arrays of C strings and pointers to them to ensure they live long enough
        let mut evnames: Vec<ffi::CString> = Vec::with_capacity(names.len());
        let mut evptrs: Vec<*const libc::c_char> = Vec::with_capacity(names.len());

        for name in &names {
            let cstr = ffi::CString::new(*name).unwrap();
            evptrs.push(cstr.as_ptr());
            evnames.push(cstr);
        }

        // shouldn't be necessary, but guards against incorrect array sizing
        evptrs.shrink_to_fit();

        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterEvalOneIter(
            self.handle,
            iteration,
            s.as_mut_ptr(),
            evptrs.as_mut_ptr(),
            dmats.len() as u64,
            &mut out_result
        ))?;
        let out = unsafe { ffi::CStr::from_ptr(out_result).to_str().unwrap().to_owned() };
        Ok(Booster::parse_eval_string(&out, &names))
    }

    /// Evaluate given matrix against this model using metrics defined in this model's parameters.
    ///
    /// See `parameter::learning::EvaluationMetric` for a full list.
    ///
    /// Returns a map of evaluation metric name to score.
    ///
    /// # Panics
    ///
    /// Will panic, if the given matrix cannot be evaluated with the given metric.
    pub fn evaluate(&self, dmat: &DMatrix, name: &str) -> XGBResult<HashMap<String, f32>> {
        let mut eval = self.eval_set(&[(dmat, name)], 0)?;
        let mut result = HashMap::new();
        eval.remove(name).unwrap().into_iter().for_each(|(k, v)| {
            result.insert(k, v);
        });

        Ok(result)
    }

    /// Get a string attribute that was previously set for this model.
    ///
    /// # Panics
    ///
    /// Will panic, if the attribute can't be retrieved, or the key can't be represented
    /// as a `CString`.
    pub fn get_attribute(&self, key: &str) -> XGBResult<Option<String>> {
        let key = ffi::CString::new(key).unwrap();
        let mut out_buf = ptr::null();
        let mut success = 0;
        xgb_call!(xgboost_sys::XGBoosterGetAttr(
            self.handle,
            key.as_ptr(),
            &mut out_buf,
            &mut success
        ))?;
        if success == 0 {
            return Ok(None);
        }
        assert!(success == 1);

        let c_str: &ffi::CStr = unsafe { ffi::CStr::from_ptr(out_buf) };
        let out = c_str.to_str().unwrap();
        Ok(Some(out.to_owned()))
    }

    /// Store a string attribute in this model with given key.
    ///
    /// # Panics
    ///
    /// Will panic, if the attribute can't be set by `XGBoost`.
    pub fn set_attribute(&mut self, key: &str, value: &str) -> XGBResult<()> {
        let key = ffi::CString::new(key).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetAttr(
            self.handle,
            key.as_ptr(),
            value.as_ptr()
        ))
    }

    /// Get names of all attributes stored in this model. Values can then be fetched with calls to `get_attribute`.
    ///
    /// # Panics
    ///
    /// Will panic, if the attribtue name cannot be retrieved from `XGBoost`.
    pub fn get_attribute_names(&self) -> XGBResult<Vec<String>> {
        let mut out_len = 0;
        let mut out = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterGetAttrNames(
            self.handle,
            &mut out_len,
            &mut out
        ))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out, out_len as usize) };
        let out_vec = out_ptr_slice
            .iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();
        Ok(out_vec)
    }

    /// This method calculates the predicions from a given matrix.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot make predictions.
    pub fn predict_from_dmat(
        &self,
        dmat: &DMatrix,
        out_shape: &[u64; 2],
        out_dim: &mut u64,
    ) -> XGBResult<Vec<f32>> {
        let json_config = "{\"type\": 0,\"training\": false,\"iteration_begin\": 0,\"iteration_end\": 0,\"strict_shape\": true}".to_string();

        let mut out_result = ptr::null();

        let c_json_config = ffi::CString::new(json_config).unwrap();

        xgb_call!(xgboost_sys::XGBoosterPredictFromDMatrix(
            self.handle,
            dmat.handle,
            c_json_config.as_ptr(),
            &mut out_shape.as_ptr(),
            out_dim,
            &mut out_result
        ))?;

        let out_len = out_shape[0];

        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict results for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    ///
    /// # Panics
    ///
    /// Will panic, if the predictions aren't possible for `XGBoost` or the results cannot be
    /// parsed.
    pub fn predict(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;

        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Predict margin for given data.
    ///
    /// Returns an array containing one entry per row in the given data.
    ///
    /// # Panics
    ///
    /// Will panic, if the predictions aren't possible for `XGBoost` or the results cannot be
    /// parsed.
    pub fn predict_margin(&self, dmat: &DMatrix) -> XGBResult<Vec<f32>> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::OutputMargin]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            1,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());
        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        Ok(data)
    }

    /// Get predicted leaf index for each sample in given data.
    ///
    /// Returns an array of shape (number of samples, number of trees) as tuple of (data, `num_rows`).
    ///
    /// Note: the leaf index of a tree is unique per tree, so e.g. leaf 1 could be found in both tree 1 and tree 0.
    ///
    /// # Panics
    ///
    /// Will panic, if the prediction of a leave isn't possible for `XGBoost` or the data cannot be
    /// parsed.
    pub fn predict_leaf(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictLeaf]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get feature contributions (SHAP values) for each prediction.
    ///
    /// The sum of all feature contributions is equal to the run untransformed margin value of the
    /// prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1) as a tuple of
    /// (data, `num_rows`). The final column contains the bias term.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot predict the data or parse the result.
    pub fn predict_contributions(&self, dmat: &DMatrix) -> XGBResult<(Vec<f32>, (usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictContribitions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();
        let num_cols = data.len() / num_rows;
        Ok((data, (num_rows, num_cols)))
    }

    /// Get SHAP interaction values for each pair of features for each prediction.
    ///
    /// The sum of each row (or column) of the interaction values equals the corresponding SHAP
    /// value (from `predict_contributions`), and the sum of the entire matrix equals the raw
    /// untransformed margin value of the prediction.
    ///
    /// Returns an array of shape (number of samples, number of features + 1, number of features + 1).
    /// The final row and column contain the bias terms.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot predict the data or parse the result.
    pub fn predict_interactions(
        &self,
        dmat: &DMatrix,
    ) -> XGBResult<(Vec<f32>, (usize, usize, usize))> {
        let option_mask = PredictOption::options_as_mask(&[PredictOption::PredictInteractions]);
        let ntree_limit = 0;
        let mut out_len = 0;
        let mut out_result = ptr::null();
        xgb_call!(xgboost_sys::XGBoosterPredict(
            self.handle,
            dmat.handle,
            option_mask,
            ntree_limit,
            0,
            &mut out_len,
            &mut out_result
        ))?;
        assert!(!out_result.is_null());

        let data = unsafe { slice::from_raw_parts(out_result, out_len as usize).to_vec() };
        let num_rows = dmat.num_rows();

        let dim = ((data.len() / num_rows) as f64).sqrt() as usize;
        Ok((data, (num_rows, dim, dim)))
    }

    /// Get a dump of this model as a string.
    ///
    /// * `with_statistics` - whether to include statistics in output dump
    /// * `feature_map` - if given, map feature IDs to feature names from given map
    pub fn dump_model(
        &self,
        with_statistics: bool,
        feature_map: Option<&FeatureMap>,
    ) -> XGBResult<String> {
        if let Some(fmap) = feature_map {
            let tmp_dir = match tempfile::tempdir() {
                Ok(dir) => dir,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            let file_path = tmp_dir.path().join("fmap.txt");
            let mut file: File = match File::create(&file_path) {
                Ok(f) => f,
                Err(err) => return Err(XGBError::new(err.to_string())),
            };

            for (feature_num, (feature_name, feature_type)) in &fmap.0 {
                writeln!(file, "{}\t{}\t{}", feature_num, feature_name, feature_type).unwrap();
            }

            self.dump_model_fmap(with_statistics, Some(&file_path))
        } else {
            self.dump_model_fmap(with_statistics, None)
        }
    }

    fn dump_model_fmap(
        &self,
        with_statistics: bool,
        feature_map_path: Option<&PathBuf>,
    ) -> XGBResult<String> {
        let fmap = if let Some(path) = feature_map_path {
            ffi::CString::new(path.as_os_str().as_bytes()).unwrap()
        } else {
            ffi::CString::new("").unwrap()
        };
        let format = ffi::CString::new("text").unwrap();
        let mut out_len = 0;
        let mut out_dump_array = ptr::null_mut();
        xgb_call!(xgboost_sys::XGBoosterDumpModelEx(
            self.handle,
            fmap.as_ptr(),
            i32::from(with_statistics),
            format.as_ptr(),
            &mut out_len,
            &mut out_dump_array
        ))?;

        let out_ptr_slice = unsafe { slice::from_raw_parts(out_dump_array, out_len as usize) };
        let out_vec: Vec<String> = out_ptr_slice
            .iter()
            .map(|str_ptr| unsafe { ffi::CStr::from_ptr(*str_ptr).to_str().unwrap().to_owned() })
            .collect();

        assert_eq!(out_len as usize, out_vec.len());
        Ok(out_vec.join("\n"))
    }

    pub(crate) fn load_rabit_checkpoint(&self) -> XGBResult<i32> {
        let mut version = 0;
        xgb_call!(xgboost_sys::XGBoosterLoadRabitCheckpoint(
            self.handle,
            &mut version
        ))?;
        Ok(version)
    }

    pub(crate) fn save_rabit_checkpoint(&self) -> XGBResult<()> {
        xgb_call!(xgboost_sys::XGBoosterSaveRabitCheckpoint(self.handle))
    }

    /// Sets the parameters for `XGBoost` from a json file.
    ///
    /// # Panics
    ///
    /// Will panic, if `XGBoost` cannot set the values.
    fn set_param_from_json(&mut self, config: HashMap<&str, &str>) {
        for (k, v) in config {
            let name = ffi::CString::new(k).unwrap();
            let value = ffi::CString::new(v).unwrap();

            unsafe { xgboost_sys::XGBoosterSetParam(self.handle, name.as_ptr(), value.as_ptr()) };
        }
    }

    fn set_param(&mut self, name: &str, value: &str) -> XGBResult<()> {
        let name = ffi::CString::new(name).unwrap();
        let value = ffi::CString::new(value).unwrap();
        xgb_call!(xgboost_sys::XGBoosterSetParam(
            self.handle,
            name.as_ptr(),
            value.as_ptr()
        ))
    }

    fn parse_eval_string(eval: &str, evnames: &[&str]) -> IndexMap<String, IndexMap<String, f32>> {
        let mut result: IndexMap<String, IndexMap<String, f32>> = IndexMap::new();

        debug!("Parsing evaluation line: {}", &eval);
        for part in eval.split('\t').skip(1) {
            for evname in evnames {
                if part.starts_with(evname) {
                    let metric_parts: Vec<&str> =
                        part[evname.len() + 1..].split(':').into_iter().collect();
                    assert_eq!(metric_parts.len(), 2);
                    let metric = metric_parts[0];
                    let score = metric_parts[1].parse::<f32>().unwrap_or_else(|_| {
                        panic!("Unable to parse XGBoost metrics output: {}", eval)
                    });

                    let metric_map = result
                        .entry(String::from(*evname))
                        .or_insert_with(IndexMap::new);
                    metric_map.insert(metric.to_owned(), score);
                }
            }
        }

        debug!("result: {:?}", &result);
        result
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        xgb_call!(xgboost_sys::XGBoosterFree(self.handle)).unwrap();
    }
}

/// Maps a feature index to a name and type, used when dumping models as text.
///
/// See [`dump_model`](struct.Booster.html#method.dump_model) for usage.
pub struct FeatureMap(BTreeMap<u32, (String, FeatureType)>);

impl FeatureMap {
    /// Read a `FeatureMap` from a file at given path.
    ///
    /// File should contain one feature definition per line, and be of the form:
    /// ```text
    /// <number>\t<name>\t<type>\n
    /// ```
    ///
    /// Type should be one of:
    /// * `i` - binary feature
    /// * `q` - quantitative feature
    /// * `int` - integer features
    ///
    /// E.g.:
    /// ```text
    /// 0   age int
    /// 1   is-parent?=yes  i
    /// 2   is-parent?=no   i
    /// 3   income  int
    /// ```
    ///
    /// # Panics
    ///
    /// Will panic, if the given `FeatureMap` file cannot be loaded.
    pub fn from_file<P: AsRef<Path>>(path: P) -> io::Result<FeatureMap> {
        let file = File::open(path)?;
        let mut features: FeatureMap = FeatureMap(BTreeMap::new());

        for (i, line) in BufReader::new(&file).lines().enumerate() {
            let line = line?;
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() != 3 {
                let msg = format!(
                    "Unable to parse features from line {}, expected 3 tab separated values",
                    i + 1
                );
                return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
            }

            assert_eq!(parts.len(), 3);
            let feature_num: u32 = match parts[0].parse() {
                Ok(num) => num,
                Err(err) => {
                    let msg = format!(
                        "Unable to parse features from line {}, could not parse feature number: {}",
                        i + 1,
                        err
                    );
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };

            let feature_name = parts[1];
            let feature_type = match FeatureType::from_str(parts[2]) {
                Ok(feature_type) => feature_type,
                Err(msg) => {
                    let msg = format!("Unable to parse features from line {}: {}", i + 1, msg);
                    return Err(io::Error::new(io::ErrorKind::InvalidData, msg));
                }
            };
            features
                .0
                .insert(feature_num, (feature_name.to_string(), feature_type));
        }
        Ok(features)
    }
}

/// Indicates the type of a feature, used when dumping models as text.
pub enum FeatureType {
    /// Binary indicator feature.
    Binary,

    /// Quantitative feature (e.g. age, time, etc.), can be missing.
    Quantitative,

    /// Integer feature (when hinted, decision boundary will be integer).
    Integer,
}

impl FromStr for FeatureType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "i" => Ok(FeatureType::Binary),
            "q" => Ok(FeatureType::Quantitative),
            "int" => Ok(FeatureType::Integer),
            _ => Err(format!(
                "unrecognised feature type '{}', must be one of: 'i', 'q', 'int'",
                s
            )),
        }
    }
}

impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = match self {
            FeatureType::Binary => "i",
            FeatureType::Quantitative => "q",
            FeatureType::Integer => "int",
        };
        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;

    use super::*;
    use crate::pro::ml::bindings::parameters::{self, learning, tree};

    fn read_train_matrix() -> XGBResult<DMatrix> {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pro/ml/bindings");
        DMatrix::load(format!("{}/data.csv?format=csv", data_path))
    }

    fn load_test_booster() -> Booster {
        let dmat = read_train_matrix().expect("Reading train matrix failed");
        Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat])
            .expect("Creating Booster failed")
    }

    #[test]
    fn set_booster_parhm() {
        let mut booster = load_test_booster();
        let res = booster.set_param("key", "value");
        assert!(res.is_ok());
    }

    #[test]
    fn load_rabit_version() {
        let version = load_test_booster().load_rabit_checkpoint().unwrap();
        assert_eq!(version, 0);
    }

    #[test]
    fn get_set_attr() {
        let mut booster = load_test_booster();
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn save_and_load_from_buffer() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pro/ml/bindings");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&BoosterParameters::default(), &[&dmat_train]).unwrap();
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, None);

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("test-xgboost-model");
        booster.save(&path).expect("saving booster failed");
        drop(booster);
        let bytes = std::fs::read(&path).expect("reading saved booster file failed");
        let booster = Booster::load_buffer(&bytes[..]).expect("loading booster from buffer failed");
        let attr = booster
            .get_attribute("foo")
            .expect("Getting attribute failed");
        assert_eq!(attr, Some("bar".to_owned()));
    }

    #[test]
    fn get_attribute_names() {
        let mut booster = load_test_booster();
        let attrs = booster
            .get_attribute_names()
            .expect("Getting attributes failed");
        assert_eq!(attrs, Vec::<String>::new());

        booster
            .set_attribute("foo", "bar")
            .expect("Setting attribute failed");
        booster
            .set_attribute("another", "another")
            .expect("Setting attribute failed");
        booster
            .set_attribute("4", "4")
            .expect("Setting attribute failed");
        booster
            .set_attribute("an even longer attribute name?", "")
            .expect("Setting attribute failed");

        let mut expected = vec!["foo", "another", "4", "an even longer attribute name?"];
        expected.sort_unstable();
        let mut attrs = booster
            .get_attribute_names()
            .expect("Getting attributes failed");
        attrs.sort();
        assert_eq!(attrs, expected);
    }

    #[test]
    fn predict() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pro/ml/bindings");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::MapCutNegative(4),
                learning::EvaluationMetric::LogLoss,
                learning::EvaluationMetric::BinaryErrorRate(0.5),
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        for i in 0..10 {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let eps = 1e-6;

        let train_metrics = booster.evaluate(&dmat_train, "default").unwrap();
        assert!(*train_metrics.get("logloss").unwrap() - 0.006_634 < eps);
        assert!(*train_metrics.get("map@4-").unwrap() - 0.001_274 < eps);

        let test_metrics = booster.evaluate(&dmat_test, "default").unwrap();
        assert!(*test_metrics.get("logloss").unwrap() - 0.006_92 < eps);
        assert!(*test_metrics.get("map@4-").unwrap() - 0.005_155 < eps);

        let v = booster.predict(&dmat_test).unwrap();
        assert_eq!(v.len(), dmat_test.num_rows());

        // first 10 predictions
        let expected_start = [
            0.005_015_169_3,
            0.988_446_7,
            0.005_015_169_3,
            0.005_015_169_3,
            0.026_636_455,
            0.117_893_63,
            0.988_446_7,
            0.012_314_71,
            0.988_446_7,
            0.000_136_560_63,
        ];

        // last 10 predictions
        let expected_end = [
            0.002_520_344,
            0.000_609_179_26,
            0.998_810_05,
            0.000_609_179_26,
            0.000_609_179_26,
            0.000_609_179_26,
            0.000_609_179_26,
            0.998_110_2,
            0.002_855_195,
            0.998_110_2,
        ];

        for (pred, expected) in v.iter().zip(&expected_start) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }

        for (pred, expected) in v[v.len() - 10..].iter().zip(&expected_end) {
            println!("predictions={}, expected={}", pred, expected);
            assert!(pred - expected < eps);
        }
    }

    #[test]
    fn predict_leaf() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pro/ml/bindings");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 15;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_leaf(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        assert_eq!(shape, (num_samples, num_rounds as usize));
    }

    #[test]
    fn predict_contributions() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pro/ml/bindings");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();

        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_contributions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1));
    }

    #[test]
    fn predict_interactions() {
        let data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/pro/ml/bindings");
        let dmat_train =
            DMatrix::load(format!("{}/agaricus.txt.train?format=libsvm", data_path)).unwrap();
        let dmat_test =
            DMatrix::load(format!("{}/agaricus.txt.test?format=libsvm", data_path)).unwrap();


        let tree_params = tree::TreeBoosterParametersBuilder::default()
            .max_depth(2)
            .eta(1.0)
            .build()
            .unwrap();
        let learning_params = learning::LearningTaskParametersBuilder::default()
            .objective(learning::Objective::BinaryLogistic)
            .eval_metrics(learning::Metrics::Custom(vec![
                learning::EvaluationMetric::LogLoss,
            ]))
            .build()
            .unwrap();
        let params = parameters::BoosterParametersBuilder::default()
            .booster_type(parameters::BoosterType::Tree(tree_params))
            .learning_params(learning_params)
            .verbose(false)
            .build()
            .unwrap();
        let mut booster =
            Booster::new_with_cached_dmats(&params, &[&dmat_train, &dmat_test]).unwrap();

        let num_rounds = 5;
        for i in 0..num_rounds {
            booster.update(&dmat_train, i).expect("update failed");
        }

        let (_preds, shape) = booster.predict_interactions(&dmat_test).unwrap();
        let num_samples = dmat_test.num_rows();
        let num_features = dmat_train.num_cols();
        assert_eq!(shape, (num_samples, num_features + 1, num_features + 1));
    }

    #[test]
    fn parse_eval_string() {
        let s = "[0]\ttrain-map@4-:0.5\ttrain-logloss:1.0\ttest-map@4-:0.25\ttest-logloss:0.75";
        let mut metrics = IndexMap::new();

        let mut train_metrics = IndexMap::new();
        train_metrics.insert("map@4-".to_owned(), 0.5);
        train_metrics.insert("logloss".to_owned(), 1.0);

        let mut test_metrics = IndexMap::new();
        test_metrics.insert("map@4-".to_owned(), 0.25);
        test_metrics.insert("logloss".to_owned(), 0.75);

        metrics.insert("train".to_owned(), train_metrics);
        metrics.insert("test".to_owned(), test_metrics);
        assert_eq!(Booster::parse_eval_string(s, &["train", "test"]), metrics);
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn pred_from_dmat() {
        let data_arr_2d = arr2(&[
            [
                8.325_200_00e+00,
                4.100_000_00e+01,
                6.984_126_98e+00,
                1.023_809_52e+00,
                3.220_000_00e+02,
                2.555_555_56e+00,
                3.788_000_00e+01,
                -1.222_300_00e+02,
            ],
            [
                8.301_400_00e+00,
                2.100_000_00e+01,
                6.238_137_08e+00,
                9.718_804_92e-01,
                2.401_000_00e+03,
                2.109_841_83e+00,
                3.786_000_00e+01,
                -1.222_200_00e+02,
            ],
            [
                7.257_400_00e+00,
                5.200_000_00e+01,
                8.288_135_59e+00,
                1.073_446_33e+00,
                4.960_000_00e+02,
                2.802_259_89e+00,
                3.785_000_00e+01,
                -1.222_400_00e+02,
            ],
            [
                5.643_100_00e+00,
                5.200_000_00e+01,
                5.817_351_60e+00,
                1.073_059_36e+00,
                5.580_000_00e+02,
                2.547_945_21e+00,
                3.785_000_00e+01,
                -1.222_500_00e+02,
            ],
            [
                3.846_200_00e+00,
                5.200_000_00e+01,
                6.281_853_28e+00,
                1.081_081_08e+00,
                5.650_000_00e+02,
                2.181_467_18e+00,
                3.785_000_00e+01,
                -1.222_500_00e+02,
            ],
            [
                4.036_800_00e+00,
                5.200_000_00e+01,
                4.761_658_03e+00,
                1.103_626_94e+00,
                4.130_000_00e+02,
                2.139_896_37e+00,
                3.785_000_00e+01,
                -1.222_500_00e+02,
            ],
            [
                3.659_100_00e+00,
                5.200_000_00e+01,
                4.931_906_61e+00,
                9.513_618_68e-01,
                1.094_000_00e+03,
                2.128_404_67e+00,
                3.784_000_00e+01,
                -1.222_500_00e+02,
            ],
            [
                3.120_000_00e+00,
                5.200_000_00e+01,
                4.797_527_05e+00,
                1.061_823_80e+00,
                1.157_000_00e+03,
                1.788_253_48e+00,
                3.784_000_00e+01,
                -1.222_500_00e+02,
            ],
            [
                2.080_400_00e+00,
                4.200_000_00e+01,
                4.294_117_65e+00,
                1.117_647_06e+00,
                1.206_000_00e+03,
                2.026_890_76e+00,
                3.784_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                3.691_200_00e+00,
                5.200_000_00e+01,
                4.970_588_24e+00,
                9.901_960_78e-01,
                1.551_000_00e+03,
                2.172_268_91e+00,
                3.784_000_00e+01,
                -1.222_500_00e+02,
            ],
            [
                3.203_100_00e+00,
                5.200_000_00e+01,
                5.477_611_94e+00,
                1.079_601_99e+00,
                9.100_000_00e+02,
                2.263_681_59e+00,
                3.785_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                3.270_500_00e+00,
                5.200_000_00e+01,
                4.772_479_56e+00,
                1.024_523_16e+00,
                1.504_000_00e+03,
                2.049_046_32e+00,
                3.785_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                3.075_000_00e+00,
                5.200_000_00e+01,
                5.322_649_57e+00,
                1.012_820_51e+00,
                1.098_000_00e+03,
                2.346_153_85e+00,
                3.785_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.673_600_00e+00,
                5.200_000_00e+01,
                4.000_000_00e+00,
                1.097_701_15e+00,
                3.450_000_00e+02,
                1.982_758_62e+00,
                3.784_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                1.916_700_00e+00,
                5.200_000_00e+01,
                4.262_903_23e+00,
                1.009_677_42e+00,
                1.212_000_00e+03,
                1.954_838_71e+00,
                3.785_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.125_000_00e+00,
                5.000_000_00e+01,
                4.242_424_24e+00,
                1.071_969_70e+00,
                6.970_000_00e+02,
                2.640_151_52e+00,
                3.785_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.775_000_00e+00,
                5.200_000_00e+01,
                5.939_577_04e+00,
                1.048_338_37e+00,
                7.930_000_00e+02,
                2.395_770_39e+00,
                3.785_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.120_200_00e+00,
                5.200_000_00e+01,
                4.052_805_28e+00,
                9.669_967_00e-01,
                6.480_000_00e+02,
                2.138_613_86e+00,
                3.785_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.991_100_00e+00,
                5.000_000_00e+01,
                5.343_675_42e+00,
                1.085_918_85e+00,
                9.900_000_00e+02,
                2.362_768_50e+00,
                3.784_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.603_300_00e+00,
                5.200_000_00e+01,
                5.465_454_55e+00,
                1.083_636_36e+00,
                6.900_000_00e+02,
                2.509_090_91e+00,
                3.784_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.357_800_00e+00,
                4.000_000_00e+01,
                4.524_096_39e+00,
                1.108_433_73e+00,
                4.090_000_00e+02,
                2.463_855_42e+00,
                3.785_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.713_500_00e+00,
                4.200_000_00e+01,
                4.478_142_08e+00,
                1.002_732_24e+00,
                9.290_000_00e+02,
                2.538_251_37e+00,
                3.785_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.725_000_00e+00,
                5.200_000_00e+01,
                5.096_234_31e+00,
                1.131_799_16e+00,
                1.015_000_00e+03,
                2.123_430_96e+00,
                3.784_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.180_600_00e+00,
                5.200_000_00e+01,
                5.193_846_15e+00,
                1.036_923_08e+00,
                8.530_000_00e+02,
                2.624_615_38e+00,
                3.784_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.600_000_00e+00,
                5.200_000_00e+01,
                5.270_142_18e+00,
                1.035_545_02e+00,
                1.006_000_00e+03,
                2.383_886_26e+00,
                3.784_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.403_800_00e+00,
                4.100_000_00e+01,
                4.495_798_32e+00,
                1.033_613_45e+00,
                3.170_000_00e+02,
                2.663_865_55e+00,
                3.785_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                2.459_700_00e+00,
                4.900_000_00e+01,
                4.728_033_47e+00,
                1.020_920_50e+00,
                6.070_000_00e+02,
                2.539_748_95e+00,
                3.785_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.808_000_00e+00,
                5.200_000_00e+01,
                4.780_856_42e+00,
                1.060_453_40e+00,
                1.102_000_00e+03,
                2.775_818_64e+00,
                3.785_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.642_400_00e+00,
                5.000_000_00e+01,
                4.401_691_33e+00,
                1.040_169_13e+00,
                1.131_000_00e+03,
                2.391_120_51e+00,
                3.784_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.687_500_00e+00,
                5.200_000_00e+01,
                4.703_225_81e+00,
                1.032_258_06e+00,
                3.950_000_00e+02,
                2.548_387_10e+00,
                3.784_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.927_400_00e+00,
                4.900_000_00e+01,
                5.068_783_07e+00,
                1.182_539_68e+00,
                8.630_000_00e+02,
                2.283_068_78e+00,
                3.784_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.961_500_00e+00,
                5.200_000_00e+01,
                4.882_086_17e+00,
                1.090_702_95e+00,
                1.168_000_00e+03,
                2.648_526_08e+00,
                3.784_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.796_900_00e+00,
                4.800_000_00e+01,
                5.737_313_43e+00,
                1.220_895_52e+00,
                1.026_000_00e+03,
                3.062_686_57e+00,
                3.784_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.375_000_00e+00,
                4.900_000_00e+01,
                5.030_395_14e+00,
                1.112_462_01e+00,
                7.540_000_00e+02,
                2.291_793_31e+00,
                3.783_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.730_300_00e+00,
                5.100_000_00e+01,
                4.972_014_93e+00,
                1.070_895_52e+00,
                1.258_000_00e+03,
                2.347_014_93e+00,
                3.783_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.486_100_00e+00,
                4.900_000_00e+01,
                4.602_272_73e+00,
                1.068_181_82e+00,
                5.700_000_00e+02,
                2.159_090_91e+00,
                3.783_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.097_200_00e+00,
                4.800_000_00e+01,
                4.807_486_63e+00,
                1.155_080_21e+00,
                9.870_000_00e+02,
                2.639_037_43e+00,
                3.783_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.410_300_00e+00,
                5.200_000_00e+01,
                3.749_379_65e+00,
                9.677_419_35e-01,
                9.010_000_00e+02,
                2.235_732_01e+00,
                3.783_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                3.480_000_00e+00,
                5.200_000_00e+01,
                4.757_281_55e+00,
                1.067_961_17e+00,
                6.890_000_00e+02,
                2.229_773_46e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.589_800_00e+00,
                5.200_000_00e+01,
                3.494_252_87e+00,
                1.027_298_85e+00,
                1.377_000_00e+03,
                1.978_448_28e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.097_800_00e+00,
                5.200_000_00e+01,
                4.215_189_87e+00,
                1.060_759_49e+00,
                9.460_000_00e+02,
                2.394_936_71e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                1.285_200_00e+00,
                5.100_000_00e+01,
                3.759_036_14e+00,
                1.248_995_98e+00,
                5.170_000_00e+02,
                2.076_305_22e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                1.025_000_00e+00,
                4.900_000_00e+01,
                3.772_486_77e+00,
                1.068_783_07e+00,
                4.620_000_00e+02,
                2.444_444_44e+00,
                3.784_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                3.964_300_00e+00,
                5.200_000_00e+01,
                4.797_979_80e+00,
                1.020_202_02e+00,
                4.670_000_00e+02,
                2.358_585_86e+00,
                3.784_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                3.012_500_00e+00,
                5.200_000_00e+01,
                4.941_780_82e+00,
                1.065_068_49e+00,
                6.600_000_00e+02,
                2.260_273_97e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.676_800_00e+00,
                5.200_000_00e+01,
                4.335_078_53e+00,
                1.099_476_44e+00,
                7.180_000_00e+02,
                1.879_581_15e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.026_000_00e+00,
                5.000_000_00e+01,
                3.700_657_89e+00,
                1.059_210_53e+00,
                6.160_000_00e+02,
                2.026_315_79e+00,
                3.783_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                1.734_800_00e+00,
                4.300_000_00e+01,
                3.980_237_15e+00,
                1.233_201_58e+00,
                5.580_000_00e+02,
                2.205_533_60e+00,
                3.782_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                9.506_000_00e-01,
                4.000_000_00e+01,
                3.900_000_00e+00,
                1.218_750_00e+00,
                4.230_000_00e+02,
                2.643_750_00e+00,
                3.782_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                1.775_000_00e+00,
                4.000_000_00e+01,
                2.687_500_00e+00,
                1.065_340_91e+00,
                7.000_000_00e+02,
                1.988_636_36e+00,
                3.782_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                9.218_000_00e-01,
                2.100_000_00e+01,
                2.045_662_10e+00,
                1.034_246_58e+00,
                7.350_000_00e+02,
                1.678_082_19e+00,
                3.782_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.504_500_00e+00,
                4.300_000_00e+01,
                4.589_680_59e+00,
                1.120_393_12e+00,
                1.061_000_00e+03,
                2.606_879_61e+00,
                3.782_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.110_800_00e+00,
                4.100_000_00e+01,
                4.473_611_11e+00,
                1.184_722_22e+00,
                1.959_000_00e+03,
                2.720_833_33e+00,
                3.782_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.247_500_00e+00,
                5.200_000_00e+01,
                4.075_000_00e+00,
                1.140_000_00e+00,
                1.162_000_00e+03,
                2.905_000_00e+00,
                3.782_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.609_800_00e+00,
                5.200_000_00e+01,
                5.021_459_23e+00,
                1.008_583_69e+00,
                7.010_000_00e+02,
                3.008_583_69e+00,
                3.782_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.411_300_00e+00,
                5.200_000_00e+01,
                4.295_454_55e+00,
                1.104_545_45e+00,
                5.760_000_00e+02,
                2.618_181_82e+00,
                3.782_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.505_700_00e+00,
                5.200_000_00e+01,
                4.779_922_78e+00,
                1.111_969_11e+00,
                6.220_000_00e+02,
                2.401_544_40e+00,
                3.782_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                8.172_000_00e-01,
                5.200_000_00e+01,
                6.102_459_02e+00,
                1.372_950_82e+00,
                7.280_000_00e+02,
                2.983_606_56e+00,
                3.782_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.217_100_00e+00,
                5.200_000_00e+01,
                4.562_500_00e+00,
                1.121_710_53e+00,
                1.074_000_00e+03,
                3.532_894_74e+00,
                3.782_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                2.562_500_00e+00,
                2.000_000_00e+00,
                2.771_929_82e+00,
                7.543_859_65e-01,
                9.400_000_00e+01,
                1.649_122_81e+00,
                3.782_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                3.392_900_00e+00,
                5.200_000_00e+01,
                5.994_652_41e+00,
                1.128_342_25e+00,
                5.540_000_00e+02,
                2.962_566_84e+00,
                3.783_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                6.118_300_00e+00,
                4.900_000_00e+01,
                5.869_565_22e+00,
                1.260_869_57e+00,
                8.600_000_00e+01,
                3.739_130_43e+00,
                3.782_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                9.011_000_00e-01,
                5.000_000_00e+01,
                6.229_508_20e+00,
                1.557_377_05e+00,
                3.770_000_00e+02,
                3.090_163_93e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                1.191_000_00e+00,
                5.200_000_00e+01,
                7.698_113_21e+00,
                1.490_566_04e+00,
                5.210_000_00e+02,
                3.276_729_56e+00,
                3.781_000_00e+01,
                -1.223_000_00e+02,
            ],
            [
                2.593_800_00e+00,
                4.800_000_00e+01,
                6.225_563_91e+00,
                1.368_421_05e+00,
                3.920_000_00e+02,
                2.947_368_42e+00,
                3.781_000_00e+01,
                -1.223_000_00e+02,
            ],
            [
                1.166_700_00e+00,
                5.200_000_00e+01,
                5.401_069_52e+00,
                1.117_647_06e+00,
                6.040_000_00e+02,
                3.229_946_52e+00,
                3.781_000_00e+01,
                -1.223_000_00e+02,
            ],
            [
                8.056_000_00e-01,
                4.800_000_00e+01,
                4.382_530_12e+00,
                1.066_265_06e+00,
                7.880_000_00e+02,
                2.373_493_98e+00,
                3.781_000_00e+01,
                -1.223_000_00e+02,
            ],
            [
                2.609_400_00e+00,
                5.200_000_00e+01,
                6.986_394_56e+00,
                1.659_863_95e+00,
                4.920_000_00e+02,
                3.346_938_78e+00,
                3.780_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                1.851_600_00e+00,
                5.200_000_00e+01,
                6.975_609_76e+00,
                1.329_268_29e+00,
                2.740_000_00e+02,
                3.341_463_41e+00,
                3.781_000_00e+01,
                -1.223_000_00e+02,
            ],
            [
                9.802_000_00e-01,
                4.600_000_00e+01,
                4.584_288_05e+00,
                1.054_009_82e+00,
                1.823_000_00e+03,
                2.983_633_39e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                1.771_900_00e+00,
                2.600_000_00e+01,
                6.047_244_09e+00,
                1.196_850_39e+00,
                3.920_000_00e+02,
                3.086_614_17e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                7.286_000_00e-01,
                4.600_000_00e+01,
                3.375_451_26e+00,
                1.072_202_17e+00,
                5.820_000_00e+02,
                2.101_083_03e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                1.750_000_00e+00,
                4.900_000_00e+01,
                5.552_631_58e+00,
                1.342_105_26e+00,
                5.600_000_00e+02,
                3.684_210_53e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                4.999_000_00e-01,
                4.600_000_00e+01,
                1.714_285_71e+00,
                5.714_285_71e-01,
                1.800_000_00e+01,
                2.571_428_57e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                2.483_000_00e+00,
                2.000_000_00e+01,
                6.278_195_49e+00,
                1.210_526_32e+00,
                2.900_000_00e+02,
                2.180_451_13e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                9.241_000_00e-01,
                1.700_000_00e+01,
                2.817_767_65e+00,
                1.052_391_80e+00,
                7.620_000_00e+02,
                1.735_763_10e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                2.446_400_00e+00,
                3.600_000_00e+01,
                5.724_950_88e+00,
                1.104_125_74e+00,
                1.236_000_00e+03,
                2.428_290_77e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.111_100_00e+00,
                1.900_000_00e+01,
                5.830_917_87e+00,
                1.173_913_04e+00,
                7.210_000_00e+02,
                3.483_091_79e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                8.026_000_00e-01,
                2.300_000_00e+01,
                5.369_230_77e+00,
                1.150_769_23e+00,
                1.054_000_00e+03,
                3.243_076_92e+00,
                3.781_000_00e+01,
                -1.222_900_00e+02,
            ],
            [
                2.011_400_00e+00,
                3.800_000_00e+01,
                4.412_903_23e+00,
                1.135_483_87e+00,
                3.440_000_00e+02,
                2.219_354_84e+00,
                3.780_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.500_000_00e+00,
                1.700_000_00e+01,
                3.197_231_83e+00,
                1.000_000_00e+00,
                6.090_000_00e+02,
                2.107_266_44e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.166_700_00e+00,
                5.200_000_00e+01,
                3.750_000_00e+00,
                1.000_000_00e+00,
                1.830_000_00e+02,
                3.267_857_14e+00,
                3.781_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.520_800_00e+00,
                5.200_000_00e+01,
                3.908_045_98e+00,
                1.114_942_53e+00,
                2.000_000_00e+02,
                2.298_850_57e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                8.075_000_00e-01,
                5.200_000_00e+01,
                2.490_322_58e+00,
                1.058_064_52e+00,
                3.460_000_00e+02,
                2.232_258_06e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                1.808_800_00e+00,
                3.500_000_00e+01,
                5.609_467_46e+00,
                1.088_757_40e+00,
                4.670_000_00e+02,
                2.763_313_61e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                2.408_300_00e+00,
                5.200_000_00e+01,
                6.721_739_13e+00,
                1.243_478_26e+00,
                3.770_000_00e+02,
                3.278_260_87e+00,
                3.781_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                9.770_000_00e-01,
                4.000_000_00e+01,
                2.315_789_47e+00,
                1.186_842_11e+00,
                5.820_000_00e+02,
                1.531_578_95e+00,
                3.781_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                7.600_000_00e-01,
                1.000_000_00e+01,
                2.651_515_15e+00,
                1.054_545_45e+00,
                5.460_000_00e+02,
                1.654_545_45e+00,
                3.781_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                9.722_000_00e-01,
                1.000_000_00e+01,
                2.692_307_69e+00,
                1.076_923_08e+00,
                1.250_000_00e+02,
                3.205_128_21e+00,
                3.780_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.243_400_00e+00,
                5.200_000_00e+01,
                2.929_411_76e+00,
                9.176_470_59e-01,
                3.960_000_00e+02,
                4.658_823_53e+00,
                3.780_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.093_800_00e+00,
                1.600_000_00e+01,
                2.745_856_35e+00,
                1.082_872_93e+00,
                8.000_000_00e+02,
                2.209_944_75e+00,
                3.780_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                8.668_000_00e-01,
                5.200_000_00e+01,
                2.443_181_82e+00,
                9.886_363_64e-01,
                9.040_000_00e+02,
                1.027_272_73e+01,
                3.780_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                7.500_000_00e-01,
                5.200_000_00e+01,
                2.823_529_41e+00,
                9.117_647_06e-01,
                1.910_000_00e+02,
                5.617_647_06e+00,
                3.780_000_00e+01,
                -1.222_800_00e+02,
            ],
            [
                2.635_400_00e+00,
                2.700_000_00e+01,
                3.493_377_48e+00,
                1.149_006_62e+00,
                7.180_000_00e+02,
                2.377_483_44e+00,
                3.779_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                1.847_700_00e+00,
                3.900_000_00e+01,
                3.672_376_87e+00,
                1.334_047_11e+00,
                1.327_000_00e+03,
                2.841_541_76e+00,
                3.780_000_00e+01,
                -1.222_700_00e+02,
            ],
            [
                2.009_600_00e+00,
                3.600_000_00e+01,
                2.294_016_36e+00,
                1.066_293_59e+00,
                3.469_000_00e+03,
                1.493_327_59e+00,
                3.780_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.834_500_00e+00,
                3.100_000_00e+01,
                3.894_915_25e+00,
                1.127_966_10e+00,
                2.048_000_00e+03,
                1.735_593_22e+00,
                3.782_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.006_200_00e+00,
                2.900_000_00e+01,
                3.681_318_68e+00,
                1.175_824_18e+00,
                2.020_000_00e+02,
                2.219_780_22e+00,
                3.781_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                1.218_500_00e+00,
                2.200_000_00e+01,
                2.945_600_00e+00,
                1.016_000_00e+00,
                2.024_000_00e+03,
                1.619_200_00e+00,
                3.782_000_00e+01,
                -1.222_600_00e+02,
            ],
            [
                2.610_400_00e+00,
                3.700_000_00e+01,
                3.707_142_86e+00,
                1.107_142_86e+00,
                1.838_000_00e+03,
                1.875_510_20e+00,
                3.782_000_00e+01,
                -1.222_600_00e+02,
            ],
        ]);

        let target_vec = [
            4.526, 3.585, 3.521, 3.413, 3.422, 2.697, 2.992, 2.414, 2.267, 2.611, 2.815, 2.418,
            2.135, 1.913, 1.592, 1.4, 1.525, 1.555, 1.587, 1.629, 1.475, 1.598, 1.139, 0.997,
            1.326, 1.075, 0.938, 1.055, 1.089, 1.32, 1.223, 1.152, 1.104, 1.049, 1.097, 0.972,
            1.045, 1.039, 1.914, 1.76, 1.554, 1.5, 1.188, 1.888, 1.844, 1.823, 1.425, 1.375, 1.875,
            1.125, 1.719, 0.938, 0.975, 1.042, 0.875, 0.831, 0.875, 0.853, 0.803, 0.6, 0.757, 0.75,
            0.861, 0.761, 0.735, 0.784, 0.844, 0.813, 0.85, 1.292, 0.825, 0.952, 0.75, 0.675,
            1.375, 1.775, 1.021, 1.083, 1.125, 1.313, 1.625, 1.125, 1.125, 1.375, 1.188, 0.982,
            1.188, 1.625, 1.375, 5.00001, 1.625, 1.375, 1.625, 1.875, 1.792, 1.3, 1.838, 1.25, 1.7,
            1.931,
        ];

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = std::mem::size_of::<f32>() * strides_ax_0;
        let byte_size_ax_1 = std::mem::size_of::<f32>() * strides_ax_1;

        // get xgboost style matrices
        let mut xg_matrix = DMatrix::from_col_major_f32(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            100,
            9,
            -1,
            f32::NAN,
        )
        .unwrap();

        // set labels
        // TODO: make more generic

        let lbls: Vec<f32> = target_vec.iter().map(|elem| *elem as f32).collect();
        xg_matrix.set_labels(lbls.as_slice()).unwrap();

        // ------------------------------------------------------
        // start training

        let mut initial_training_config: HashMap<&str, &str> = HashMap::new();

        initial_training_config.insert("validate_parameters", "1");
        initial_training_config.insert("process_type", "default");
        initial_training_config.insert("tree_method", "hist");
        initial_training_config.insert("eval_metric", "rmse");
        initial_training_config.insert("max_depth", "3");

        let evals = &[(&xg_matrix, "train")];
        let bst = Booster::train(
            Some(evals),
            &xg_matrix,
            initial_training_config,
            None, // <- No old model yet
        )
        .unwrap();

        let test_data_arr_2d = arr2(&[
            [
                1.910_000_00e+00,
                4.600_000_00e+01,
                5.000_000_00e+00,
                1.004_132_23e+00,
                5.230_000_00e+02,
                2.161_157_02e+00,
                3.936_000_00e+01,
                -1.217_000_00e+02,
                6.390_000_00e-01,
            ],
            [
                2.047_400_00e+00,
                3.700_000_00e+01,
                4.957_446_81e+00,
                1.053_191_49e+00,
                1.505_000_00e+03,
                3.202_127_66e+00,
                3.936_000_00e+01,
                -1.217_000_00e+02,
                5.600_000_00e-01,
            ],
            [
                1.835_500_00e+00,
                3.400_000_00e+01,
                5.103_030_30e+00,
                1.127_272_73e+00,
                6.350_000_00e+02,
                3.848_484_85e+00,
                3.936_000_00e+01,
                -1.216_900_00e+02,
                6.300_000_00e-01,
            ],
            [
                2.324_300_00e+00,
                2.700_000_00e+01,
                6.347_188_26e+00,
                1.063_569_68e+00,
                1.100_000_00e+03,
                2.689_486_55e+00,
                3.938_000_00e+01,
                -1.217_400_00e+02,
                8.550_000_00e-01,
            ],
            [
                2.525_900_00e+00,
                3.000_000_00e+01,
                5.508_108_11e+00,
                1.037_837_84e+00,
                5.010_000_00e+02,
                2.708_108_11e+00,
                3.933_000_00e+01,
                -1.218_000_00e+02,
                8.130_000_00e-01,
            ],
            [
                2.281_300_00e+00,
                2.100_000_00e+01,
                5.207_272_73e+00,
                1.032_727_27e+00,
                8.620_000_00e+02,
                3.134_545_45e+00,
                3.942_000_00e+01,
                -1.217_100_00e+02,
                5.760_000_00e-01,
            ],
            [
                2.172_800_00e+00,
                2.200_000_00e+01,
                5.616_099_07e+00,
                1.058_823_53e+00,
                9.410_000_00e+02,
                2.913_312_69e+00,
                3.941_000_00e+01,
                -1.217_100_00e+02,
                5.940_000_00e-01,
            ],
            [
                2.494_300_00e+00,
                2.900_000_00e+01,
                5.050_898_20e+00,
                9.790_419_16e-01,
                8.640_000_00e+02,
                2.586_826_35e+00,
                3.940_000_00e+01,
                -1.217_500_00e+02,
                8.190_000_00e-01,
            ],
            [
                3.392_900_00e+00,
                3.900_000_00e+01,
                6.656_626_51e+00,
                1.084_337_35e+00,
                4.080_000_00e+02,
                2.457_831_33e+00,
                3.948_000_00e+01,
                -1.217_900_00e+02,
                8.210_000_00e-01,
            ],
            [
                2.381_600_00e+00,
                1.600_000_00e+01,
                6.055_954_09e+00,
                1.120_516_50e+00,
                1.516_000_00e+03,
                2.175_035_87e+00,
                3.815_000_00e+01,
                -1.204_600_00e+02,
                1.160_000_00e+00,
            ],
            [
                2.500_000_00e+00,
                1.000_000_00e+01,
                5.381_443_30e+00,
                1.116_838_49e+00,
                7.850_000_00e+02,
                2.697_594_50e+00,
                3.812_000_00e+01,
                -1.205_500_00e+02,
                1.161_000_00e+00,
            ],
            [
                2.365_400_00e+00,
                3.400_000_00e+01,
                5.590_631_36e+00,
                1.138_492_87e+00,
                1.150_000_00e+03,
                2.342_158_86e+00,
                3.809_000_00e+01,
                -1.205_600_00e+02,
                9.490_000_00e-01,
            ],
            [
                2.906_300_00e+00,
                2.700_000_00e+01,
                6.025_125_63e+00,
                1.125_628_14e+00,
                4.630_000_00e+02,
                2.326_633_17e+00,
                3.807_000_00e+01,
                -1.205_500_00e+02,
                9.220_000_00e-01,
            ],
            [
                2.287_500_00e+00,
                3.700_000_00e+01,
                5.257_142_86e+00,
                1.057_142_86e+00,
                3.390_000_00e+02,
                2.421_428_57e+00,
                3.807_000_00e+01,
                -1.205_400_00e+02,
                7.990_000_00e-01,
            ],
            [
                2.652_800_00e+00,
                9.000_000_00e+00,
                8.010_752_69e+00,
                1.586_021_51e+00,
                2.233_000_00e+03,
                2.401_075_27e+00,
                3.797_000_00e+01,
                -1.206_700_00e+02,
                1.330_000_00e+00,
            ],
            [
                3.000_000_00e+00,
                1.600_000_00e+01,
                6.110_569_11e+00,
                1.162_601_63e+00,
                1.777_000_00e+03,
                2.889_430_89e+00,
                3.809_000_00e+01,
                -1.204_600_00e+02,
                1.226_000_00e+00,
            ],
            [
                2.982_100_00e+00,
                1.900_000_00e+01,
                5.278_947_37e+00,
                1.236_842_11e+00,
                5.380_000_00e+02,
                2.831_578_95e+00,
                3.824_000_00e+01,
                -1.207_900_00e+02,
                9.040_000_00e-01,
            ],
            [
                2.047_200_00e+00,
                1.600_000_00e+01,
                5.931_558_94e+00,
                1.218_631_18e+00,
                1.319_000_00e+03,
                2.507_604_56e+00,
                3.820_000_00e+01,
                -1.209_000_00e+02,
                9.320_000_00e-01,
            ],
            [
                4.010_900_00e+00,
                8.000_000_00e+00,
                5.574_175_82e+00,
                1.063_186_81e+00,
                1.000_000_00e+03,
                2.747_252_75e+00,
                3.816_000_00e+01,
                -1.208_800_00e+02,
                1.259_000_00e+00,
            ],
            [
                3.636_000_00e+00,
                9.000_000_00e+00,
                5.994_983_28e+00,
                1.137_123_75e+00,
                1.800_000_00e+03,
                3.010_033_44e+00,
                3.811_000_00e+01,
                -1.209_100_00e+02,
                1.331_000_00e+00,
            ],
        ]);

        let strides_ax_0 = test_data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = test_data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = std::mem::size_of::<f32>() * strides_ax_0;
        let byte_size_ax_1 = std::mem::size_of::<f32>() * strides_ax_1;

        // get xgboost style matrices
        let test_data = DMatrix::from_col_major_f32(
            test_data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            20,
            9,
            -1,
            f32::NAN,
        )
        .unwrap();

        let mut out_dim: u64 = 10;
        let result = bst
            .predict_from_dmat(&test_data, &[20, 9], &mut out_dim)
            .unwrap();
        println!("result: {:?}", result);
    }
}
