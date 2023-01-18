use futures::future::BoxFuture;
use geoengine_datatypes::primitives::MachineLearningQueryRectangle;
use serde_json::Value;
use tracing::{span, Level};

use crate::engine::{MachineLearningModelQueryProcessor, QueryContext};
use crate::util::Result;

use super::abortable_query_execution;

/// This method takes a machine learning query processor and trains a model based on the underlying raster data.
pub async fn raster_stream_to_ml_model<C: QueryContext + 'static>(
    // file_path: &Path,
    processor: Box<dyn MachineLearningModelQueryProcessor<OutputFormat = Value>>,
    query_rect: MachineLearningQueryRectangle,
    mut query_ctx: C,
    conn_closed: BoxFuture<'_, ()>,
) -> Result<Value> {
    let span = span!(Level::TRACE, "raster_stream_to_ml_model");
    let _enter = span.enter();

    let query_abort_trigger = query_ctx.abort_trigger()?;

    let model = processor.model_query(query_rect, &query_ctx);

    let result = abortable_query_execution(model, conn_closed, query_abort_trigger).await?;

    Ok(result)
}

#[cfg(test)]
mod tests {

    use std::{collections::HashMap, path::PathBuf};

    use geoengine_datatypes::{
        primitives::{BoundingBox2D, DateTime, QueryRectangle, SpatialResolution, TimeInterval},
        raster::TilingSpecification,
        util::test::TestDefault,
    };
    use serial_test::serial;

    use crate::{
        engine::{MachineLearningOperator, MockExecutionContext, MockQueryContext, RasterOperator},
        util::helper::generate_raster_test_data_band_helper,
    };

    use crate::pro::{XgboostTrainingOperator, XgboostTrainingParams};

    use super::*;

    /// make sure, that a model can be generated from a raster stream.
    #[serial]
    #[tokio::test]
    async fn ml_model_from_stream() {
        let tile_size_in_pixels = [4, 2].into();
        let tiling_specification = TilingSpecification {
            origin_coordinate: [0.0, 0.0].into(),
            tile_size_in_pixels,
        };
        let ctx = MockExecutionContext::new_with_tiling_spec(tiling_specification);

        let qry_ctx = MockQueryContext::test_default();

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

        let src_a = crate::util::helper::generate_raster_test_data_band_helper(vec![
            1, 2, 3, 4, 5, 6, 7, 8,
        ]);
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

        let op: Box<dyn MachineLearningOperator> = xg_train.boxed();

        let initialized = op.initialize(&ctx).await.unwrap();

        let processor = initialized.query_processor().unwrap();
        let query_processor = processor.json_plain();

        let model = raster_stream_to_ml_model(
            query_processor,
            qry,
            qry_ctx,
            Box::pin(futures::future::pending()),
        )
        .await
        .unwrap();

        assert_eq!(
            include_bytes!("../../../test_data/pro/ml/xgboost/test_model.json") as &[u8],
            model.to_string().as_bytes()
        );
    }
}
