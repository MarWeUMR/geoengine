use geoengine_datatypes::collections::FeatureCollectionInfos;
use geoengine_datatypes::primitives::AxisAlignedRectangle;
use num_traits::AsPrimitive;
use ordered_float::NotNan;
use serde_json::Value;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::path::PathBuf;
use xgboost_rs::{Booster, DMatrix};

use async_trait::async_trait;
use futures::StreamExt;
use geoengine_datatypes::primitives::{
    partitions_extent, time_interval_extent, BoundingBox2D, MachineLearningQueryRectangle,
};
use geoengine_datatypes::pro::MachineLearningFeature;
use geoengine_datatypes::raster::GridOrEmpty;
use serde::{Deserialize, Serialize};
use snafu::{ensure, ResultExt};

use crate::engine::{
    CreateSpan, ExecutionContext, InitializedMachineLearningOperator, InitializedRasterOperator,
    InitializedVectorOperator, MachineLearningModelQueryProcessor, MachineLearningOperator,
    MachineLearningResultDescriptor, MultipleRasterOrSingleVectorSource, Operator, OperatorName,
    QueryContext, QueryProcessor, TypedMachineLearningModelQueryProcessor,
    TypedRasterQueryProcessor, TypedVectorQueryProcessor,
};
use crate::error::Error;
use crate::pro::xg_error::error as XgModuleError;
use crate::util::input::MultiRasterOrVectorOperator;
use crate::util::Result;

use tracing::{span, Level};

pub const ML_OUTPUT_TYPE: &str = "Json";
const BATCH_SIZE: usize = 1_000;

pub type XgboostTrainingOperator =
    Operator<XgboostTrainingParams, MultipleRasterOrSingleVectorSource>;

impl OperatorName for XgboostTrainingOperator {
    const TYPE_NAME: &'static str = "XgboostTrainingOperator";
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct XgboostTrainingParams {
    // where to store the model file
    pub model_store_path: Option<PathBuf>,
    pub no_data_value: f32, // FIXME: remove?
    pub training_config: HashMap<String, String>,
    pub feature_names: Vec<Option<String>>,
}

#[typetag::serde]
#[async_trait]
impl MachineLearningOperator for XgboostTrainingOperator {
    async fn _initialize(
        self: Box<Self>,
        context: &dyn ExecutionContext,
    ) -> Result<Box<dyn InitializedMachineLearningOperator>> {
        match self.sources.source {
            MultiRasterOrVectorOperator::Raster(raster_sources) => {
                ensure!(
                    (1..=8).contains(&raster_sources.len()),
                    crate::error::InvalidNumberOfRasterInputs {
                        expected: 1..8,
                        found: raster_sources.len()
                    }
                );
                ensure!( self.params.feature_names.is_empty() || self.params.feature_names.len() == raster_sources.len(),
                    crate::error::InvalidOperatorSpec {
                        reason: "XGBoost on raster data must either contain a name/alias for every feature ('feature_names' parameter) or no names at all."
                            .to_string(),
                });

                let initialized = futures::future::join_all(
                    raster_sources.into_iter().map(|op| op.initialize(context)),
                )
                .await
                .into_iter()
                .collect::<Result<Vec<_>>>()?;

                if initialized.len() > 1 {
                    let srs = initialized[0].result_descriptor().spatial_reference;
                    ensure!(
                        initialized
                            .iter()
                            .all(|op| op.result_descriptor().spatial_reference == srs),
                        crate::error::AllSourcesMustHaveSameSpatialReference
                    );
                }

                let in_descriptors = initialized
                    .iter()
                    .map(InitializedRasterOperator::result_descriptor)
                    .collect::<Vec<_>>();

                let time = time_interval_extent(in_descriptors.iter().map(|d| d.time));
                let bbox = partitions_extent(in_descriptors.iter().map(|d| d.bbox));

                Ok(InitializedXgboostTrainingOperator::new(
                    MachineLearningResultDescriptor {
                        spatial_reference: in_descriptors[0].spatial_reference,
                        time,
                        // converting `SpatialPartition2D` to `BoundingBox2D` is ok here, because is makes the covered area only larger
                        bbox: bbox
                            .and_then(|p| BoundingBox2D::new(p.lower_left(), p.upper_right()).ok()),
                    },
                    initialized,
                    self.params.model_store_path,
                    self.params.training_config,
                    self.params.feature_names,
                )
                .boxed())
            }
            MultiRasterOrVectorOperator::Vector(vector_source) => {
                ensure!( !self.params.feature_names.is_empty(),
                    crate::error::InvalidOperatorSpec {
                        reason: "XGBoost on vector data requires the selection of at least one numeric column ('feature_names' parameter)."
                            .to_string(),
                    }
                );

                let source = vector_source.initialize(context).await?;
                let in_desc = source.result_descriptor();

                Ok(InitializedXgboostTrainingOperator::new(
                    MachineLearningResultDescriptor {
                        spatial_reference: in_desc.spatial_reference,
                        time: in_desc.time,
                        bbox: in_desc.bbox,
                    },
                    source,
                    self.params.model_store_path,
                    self.params.training_config,
                    self.params.feature_names,
                )
                .boxed())
            }
        }
    }

    fn get_model_store_path(&self) -> Option<PathBuf> {
        self.params.model_store_path.clone()
    }

    span_fn!(XgboostTrainingOperator);
}

pub struct InitializedXgboostTrainingOperator<Op> {
    pub model_store_path: Option<PathBuf>,
    training_config: HashMap<String, String>,
    result_descriptor: MachineLearningResultDescriptor,
    source: Op,
    feature_names: Vec<Option<String>>,
}

impl<Op> InitializedXgboostTrainingOperator<Op> {
    pub fn new(
        result_descriptor: MachineLearningResultDescriptor,
        source: Op,
        model_store_path: Option<PathBuf>,
        training_config: HashMap<String, String>,
        feature_names: Vec<Option<String>>,
    ) -> Self {
        Self {
            model_store_path,
            training_config,
            result_descriptor,
            source,
            feature_names,
        }
    }
}

impl InitializedMachineLearningOperator
    for InitializedXgboostTrainingOperator<Box<dyn InitializedVectorOperator>>
{
    fn result_descriptor(&self) -> &MachineLearningResultDescriptor {
        &self.result_descriptor
    }

    fn query_processor(&self) -> Result<TypedMachineLearningModelQueryProcessor> {
        let processor = XgboostTrainingVectorQueryProcessor {
            input_vqp: self.source.query_processor()?,
            training_config: self.training_config.clone(),
            feature_names: self.feature_names.clone(),
        };

        Ok(TypedMachineLearningModelQueryProcessor::JsonPlain(
            processor.boxed(),
        ))
    }
}

impl InitializedMachineLearningOperator
    for InitializedXgboostTrainingOperator<Vec<Box<dyn InitializedRasterOperator>>>
{
    fn result_descriptor(&self) -> &MachineLearningResultDescriptor {
        &self.result_descriptor
    }

    fn query_processor(&self) -> Result<TypedMachineLearningModelQueryProcessor> {
        let input = self
            .source
            .iter()
            .map(InitializedRasterOperator::query_processor)
            .collect::<Result<Vec<_>>>()?;

        let processor = XgboostTrainingRasterQueryProcessor {
            input_rqp: input,
            training_config: self.training_config.clone(),
            feature_names: self.feature_names.clone(),
        };

        Ok(TypedMachineLearningModelQueryProcessor::JsonPlain(
            processor.boxed(),
        ))
    }

    fn boxed(self) -> Box<dyn InitializedMachineLearningOperator> {
        Box::new(self)
    }
}

pub struct XgboostTrainingVectorQueryProcessor {
    input_vqp: TypedVectorQueryProcessor,
    training_config: HashMap<String, String>,
    feature_names: Vec<Option<String>>,
}

#[async_trait]
impl MachineLearningModelQueryProcessor for XgboostTrainingVectorQueryProcessor {
    type OutputFormat = serde_json::Value;

    fn model_type(&self) -> &'static str {
        "Json"
    }

    async fn model_query<'a>(
        &'a self,
        query: MachineLearningQueryRectangle,
        ctx: &'a dyn QueryContext,
    ) -> Result<Self::OutputFormat, crate::error::Error> {
        // `accums` contains the column-wise data in the end
        let mut accums: Vec<MachineLearningAccum> = self
            .feature_names
            .iter()
            .map(|feature_name| {
                let accum = MachineLearningAccum::new(
                    feature_name
                        .clone()
                        .ok_or_else(|| Error::MachineLearningFeaturesNotAvailable)?,
                );
                Ok(accum)
            })
            .collect::<Result<Vec<_>>>()?;

        call_on_generic_vector_processor!(&self.input_vqp, processor => {
            let mut query = processor.query(query, ctx).await?;
            while let Some(collection) = query.next().await {
                let collection = collection?;

                for accum in &mut accums {
                    let feature_data = collection.data(&accum.feature_name).expect("checked in param");
                    let iter = feature_data.float_options_iter().map(|o| match o {
                        Some(v) => v as f32,
                        None => f32::NAN,
                    });
                    accum.update(iter);
                }
            }
        });

        let mut results = accums
            .into_iter()
            .map(|feature| match feature.accum {
                MachineLearningAccumKind::XGBoost(data) => Ok(MachineLearningFeature::new(
                    Some(feature.feature_name),
                    data,
                )),
            })
            .collect::<Vec<Result<MachineLearningFeature>>>();

        train_model(&mut results, self.training_config.clone())
    }
}

pub struct XgboostTrainingRasterQueryProcessor {
    input_rqp: Vec<TypedRasterQueryProcessor>,
    training_config: HashMap<String, String>,
    feature_names: Vec<Option<String>>,
}

impl XgboostTrainingRasterQueryProcessor {
    async fn process_raster(
        name: String,
        input_rpq: &TypedRasterQueryProcessor,
        query: MachineLearningQueryRectangle,
        ctx: &dyn QueryContext,
    ) -> Result<MachineLearningFeature> {
        call_on_generic_raster_processor!(input_rpq, processor => {

            let mut stream = processor.query(query.into(), ctx).await?;
            let mut accum = MachineLearningAccum::new(name);


            while let Some(tile) = stream.next().await {
                let tile = tile?;

                match tile.grid_array {
                    // Ignore empty grids if no_data should not be included
                    GridOrEmpty::Empty(_) => {},
                    GridOrEmpty::Grid(grid) => {
                        accum.update(
                            grid.masked_element_deref_iterator().filter_map(|pixel_option| {
                                pixel_option.map(|p| { let v: f32 = p.as_(); v})
                            }));
                    }
                }
            }

            Ok(accum.finish())
        })
    }
}

#[async_trait]
impl MachineLearningModelQueryProcessor for XgboostTrainingRasterQueryProcessor {
    type OutputFormat = serde_json::Value;

    fn model_type(&self) -> &'static str {
        ML_OUTPUT_TYPE
    }

    /// Trains a xgboost model and returns the resulting model as a serde_json::Value.
    /// This is the central method, where the xgboost dmatrix structure is created
    /// and passed to the xgboost training procedure.
    async fn model_query<'xt_rqp>(
        &'xt_rqp self,
        query: MachineLearningQueryRectangle,
        ctx: &'xt_rqp dyn QueryContext,
    ) -> Result<Self::OutputFormat> {
        let feature_names = self
            .feature_names
            .iter()
            .map(|feature| {
                feature
                    .clone()
                    .ok_or_else(|| Error::MachineLearningFeaturesNotAvailable)
            })
            .collect::<Result<Vec<String>>>()?;

        let results: Vec<_> = self
            .input_rqp
            .iter()
            .zip(feature_names.iter())
            .map(|(proc, name)| Self::process_raster(name.clone(), proc, query, ctx))
            .collect();

        let mut results: Vec<_> = futures::future::join_all(results).await;

        train_model(&mut results, self.training_config.clone())
    }
}

fn train_model(
    results: &mut Vec<Result<MachineLearningFeature>>,
    training_config: HashMap<String, String>,
) -> Result<Value> {
    ensure!(
        results.len() >= 2,
        crate::error::MachineLearningMustHaveAtLeastTwoFeatures
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
                .map_err(|_| Error::MachineLearningFeatureDataNotAvailable)?
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
    .context(XgModuleError::CreateDMatrix)?;

    let lbls_remap = remap_labels(lbls?.feature_data)?;

    dmatrix
        .set_labels(lbls_remap.as_slice())
        .context(XgModuleError::DMatrixSetLabels)?; // <- here we need the remapped target values

    let evals = &[(&dmatrix, "train")];
    let bst = Booster::train(
        Some(evals),
        &dmatrix,
        training_config,
        None, // <- No old model yet
    )
    .context(XgModuleError::BoosterTraining)?;

    let model = bst
        .save_to_buffer("json".into())
        .context(XgModuleError::ModelStorage)?;

    let res = serde_json::from_str(model.as_str())?;
    Ok(res)
}

//
// AUX Structures
//

#[derive(Debug)]
enum MachineLearningAccumKind {
    XGBoost(Vec<f32>),
}

impl MachineLearningAccumKind {
    fn update(&mut self, values: impl Iterator<Item = f32>) {
        match self {
            Self::XGBoost(ref mut x) => {
                x.extend(values.filter(|x| x.is_finite()));
            }
        }
    }
}

#[derive(Debug)]
/// Used to gather the source data before further processing/shaping for xgboost
struct MachineLearningAccum {
    feature_name: String,
    accum: MachineLearningAccumKind,
}

impl MachineLearningAccum {
    fn new(feature_name: String) -> MachineLearningAccum {
        MachineLearningAccum {
            feature_name,
            accum: MachineLearningAccumKind::XGBoost(Vec::new()),
        }
    }

    fn update(&mut self, values: impl Iterator<Item = f32>) {
        for chunk in &itertools::Itertools::chunks(values, BATCH_SIZE) {
            self.accum.update(chunk);
        }
    }

    fn finish(&mut self) -> MachineLearningFeature {
        match &self.accum {
            MachineLearningAccumKind::XGBoost(data) => {
                MachineLearningFeature::new(Some(self.feature_name.clone()), data.clone())
            }
        }
    }
}

/// This function is used to remap the values of the target band to a minimal integer set.
/// I.e.: A set of for example [12.0, 3.0, 3.0, 7.0, 4.0] would be mapped to (order doesn't matter):
/// {0.0:12.0, 1.0:3.0, 2.0:7.0, 3.0:4.0 }
fn remap_labels(lbls: Vec<f32>) -> Result<Vec<f32>> {
    let mut unique_lbl_counter: f32 = -1.0;

    // TODO: persist this hashmap
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
                    .ok_or_else(|| Error::CouldNotGetMlLabelKeyName)?;
                Ok(*remapped_val)
            }
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(remapped_values)
}

#[cfg(test)]
mod tests {

    use serial_test::serial;
    use std::collections::HashMap;
    use std::path::PathBuf;

    use crate::engine::RasterOperator;
    use crate::engine::{
        ChunkByteSize, MachineLearningOperator, MockExecutionContext, MockQueryContext,
    };
    use crate::pro::{XgboostTrainingOperator, XgboostTrainingParams};
    use crate::util::helper::{
        generate_raster_test_data_band_helper, generate_vector_test_data_band_helper,
    };

    use geoengine_datatypes::primitives::{
        BoundingBox2D, DateTime, MachineLearningQueryRectangle, QueryRectangle, SpatialResolution,
        TimeInterval, VectorQueryRectangle,
    };

    use geoengine_datatypes::raster::TilingSpecification;
    use geoengine_datatypes::util::test::TestDefault;

    #[serial]
    #[tokio::test]
    async fn test_training_on_raster_data() {
        let tile_size_in_pixels = [4, 2].into();
        let tiling_specification = TilingSpecification {
            origin_coordinate: [0.0, 0.0].into(),
            tile_size_in_pixels,
        };
        let execution_context = MockExecutionContext::new_with_tiling_spec(tiling_specification);

        let src_a = generate_raster_test_data_band_helper(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let src_b = generate_raster_test_data_band_helper(vec![9, 10, 11, 12, 13, 14, 15, 16]);
        let src_target = generate_raster_test_data_band_helper(vec![0, 1, 2, 2, 2, 1, 0, 0]);

        let mut training_config: HashMap<String, String> = HashMap::new();
        training_config.insert("validate_parameters".into(), "1".into());
        training_config.insert("process_type".into(), "default".into());
        training_config.insert("tree_method".into(), "hist".into());
        training_config.insert("max_depth".into(), "10".into());
        training_config.insert("objective".into(), "multi:softmax".into());
        training_config.insert("num_class".into(), "4".into());
        training_config.insert("eta".into(), "0.75".into());

        let xg_train = XgboostTrainingOperator {
            params: XgboostTrainingParams {
                model_store_path: Some(PathBuf::from("some_model.json")),
                no_data_value: -1_000.,
                training_config,
                feature_names: vec![Some("a".into()), Some("b".into()), Some("target".into())],
            },
            sources: vec![
                src_a.expect("Source (a) should be setup!").boxed(),
                src_b.expect("Source (b) should be setup!").boxed(),
                src_target
                    .expect("Source (target) should be setup!")
                    .boxed(),
            ]
            .into(),
        };

        let query_processor = xg_train
            .boxed()
            .initialize(&execution_context)
            .await
            .unwrap()
            .query_processor()
            .unwrap()
            .json_plain();

        let spatial_bounds: BoundingBox2D =
            BoundingBox2D::new((-180., -90.).into(), (180., 90.).into()).unwrap();

        let time_interval =
            TimeInterval::new_instant(DateTime::new_utc(2013, 12, 1, 12, 0, 0)).unwrap();

        let spatial_resolution = SpatialResolution::one();

        let qry: QueryRectangle<BoundingBox2D> = MachineLearningQueryRectangle {
            spatial_bounds,
            time_interval,
            spatial_resolution,
        };

        let model = query_processor
            .model_query(qry, &MockQueryContext::test_default())
            .await
            .unwrap();

        // check that the returned model is as expected
        assert_eq!(
            include_bytes!("../../../../test_data/pro/ml/xgboost/test_model.json") as &[u8],
            model.to_string().as_bytes()
        );
    }

    #[serial]
    #[tokio::test]
    async fn test_training_on_vector_data() {
        let vector_source = generate_vector_test_data_band_helper();

        let mut training_config: HashMap<String, String> = HashMap::new();
        training_config.insert("validate_parameters".into(), "1".into());
        training_config.insert("process_type".into(), "default".into());
        training_config.insert("tree_method".into(), "hist".into());
        training_config.insert("max_depth".into(), "10".into());
        training_config.insert("objective".into(), "multi:softmax".into());
        training_config.insert("num_class".into(), "4".into());
        training_config.insert("eta".into(), "0.75".into());

        let xg_train = XgboostTrainingOperator {
            params: XgboostTrainingParams {
                model_store_path: Some(PathBuf::from("some_model.json")),
                no_data_value: -1_000.,
                training_config,
                feature_names: vec![
                    Some("temp".to_string()),
                    Some("precipitation".to_string()),
                    Some("target".into()),
                ],
            },
            sources: vector_source.unwrap().into(),
        };

        let execution_context = MockExecutionContext::test_default();

        let query_processor = xg_train
            .boxed()
            .initialize(&execution_context)
            .await
            .unwrap()
            .query_processor()
            .unwrap()
            .json_plain();

        let model = query_processor
            .model_query(
                VectorQueryRectangle {
                    spatial_bounds: BoundingBox2D::new((-180., -90.).into(), (180., 90.).into())
                        .unwrap(),
                    time_interval: TimeInterval::default(),
                    spatial_resolution: SpatialResolution::one(),
                },
                &MockQueryContext::new(ChunkByteSize::MIN),
            )
            .await
            .unwrap();

        // check that the returned model is as expected
        assert_eq!(
            include_bytes!("../../../../test_data/pro/ml/xgboost/test_model.json") as &[u8],
            model.to_string().as_bytes()
        );
    }
}
