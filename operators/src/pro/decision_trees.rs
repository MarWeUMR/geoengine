#[cfg(test)]
mod tests {

    use crate::engine::RasterQueryProcessor;
    use crate::processing::meteosat::{
        new_channel_key, new_offset_key, new_satellite_key, new_slope_key,
    };
    use crate::util::stream_zip::{StreamTupleZip, StreamVectorZip};
    use crate::util::{create_rayon_thread_pool, spawn_blocking_with_thread_pool};
    use crate::{
        engine::{MockExecutionContext, MockQueryContext, RasterOperator, RasterResultDescriptor},
        source::{
            FileNotFoundHandling, GdalDatasetGeoTransform, GdalDatasetParameters,
            GdalMetaDataRegular, GdalMetadataMapping, GdalSource, GdalSourceParameters,
            GdalSourceTimePlaceholder, TimeReference,
        },
    };

    use futures::StreamExt;
    use geo_dtrees::tangram::tangram_wrapper::{tangram_predict, tangram_train_model, ModelType};
    use geo_dtrees::util::data_processing::{
        get_tangram_matrix, get_train_test_split_arrays, get_xg_matrix, xg_set_ground_truth,
    };
    use geoengine_datatypes::primitives::{Coordinate2D, QueryRectangle, SpatialPartition2D};
    use geoengine_datatypes::raster::{GridOrEmpty, TilingSpecification};
    use geoengine_datatypes::test_data;
    use geoengine_datatypes::util::test::TestDefault;
    use geoengine_datatypes::{
        dataset::{DatasetId, InternalDatasetId},
        hashmap,
        primitives::{
            Measurement, SpatialResolution, TimeGranularity, TimeInstance, TimeInterval, TimeStep,
        },
        raster::RasterDataType,
        spatial_reference::{SpatialReference, SpatialReferenceAuthority, SpatialReferenceOption},
    };
    use geoengine_datatypes::{raster::RasterPropertiesEntryType, util::Identifier};
    use itertools::izip;
    use ndarray::{arr2, Array, Array2};
    use rand::distributions::Uniform;
    use rand::prelude::Distribution;
    use xgboost_bindings::{parameters, Booster, DMatrix};

    use std::mem;
    use std::path::PathBuf;

    fn get_gdal_config_metadata(path: &str) -> GdalMetaDataRegular {
        let no_data_value = Some(-339999995214436424907732413799364296704.0);

        let gdal_config_metadata = GdalMetaDataRegular {
            result_descriptor: RasterResultDescriptor {
                data_type: RasterDataType::U8,
                spatial_reference: SpatialReferenceOption::SpatialReference(SpatialReference::new(
                    SpatialReferenceAuthority::Epsg,
                    32617,
                )),
                measurement: Measurement::Classification {
                    measurement: "raw".into(),
                    classes: hashmap!(0 => "Water Bodies".to_string()),
                },
                no_data_value,
            },
            params: GdalDatasetParameters {
                file_path: PathBuf::from(test_data!(path)),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform {
                    origin_coordinate: (677770.0, 4764070.0).into(),
                    x_pixel_size: 10.0,
                    y_pixel_size: -10.0,
                },
                width: 210,
                height: 170,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![
                    GdalMetadataMapping::identity(
                        new_offset_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                    GdalMetadataMapping::identity(
                        new_slope_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                    GdalMetadataMapping::identity(
                        new_channel_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                    GdalMetadataMapping::identity(
                        new_satellite_key(),
                        RasterPropertiesEntryType::Number,
                    ),
                ]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_087___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep {
                granularity: TimeGranularity::Minutes,
                step: 15,
            },
        };

        gdal_config_metadata
    }

    async fn initialize_operator(
        gcm: GdalMetaDataRegular,
    ) -> Box<dyn RasterQueryProcessor<RasterType = u8>> {
        let tiling_specification =
            TilingSpecification::new(Coordinate2D::default(), [16, 16].into());

        let mut mc = MockExecutionContext::test_default();
        mc.tiling_specification = tiling_specification;

        let id = DatasetId::Internal {
            dataset_id: InternalDatasetId::new(),
        };

        let op = Box::new(GdalSource {
            params: GdalSourceParameters {
                dataset: id.clone(),
            },
        });

        // let gcm = get_gdal_config_metadata();

        mc.add_meta_data(id, Box::new(gcm.clone()));

        let rqp_gt = op
            .initialize(&mc)
            .await
            .unwrap()
            .query_processor()
            .unwrap()
            .get_u8()
            .unwrap();

        rqp_gt
    }

    async fn get_band_data(
        rqp_gt: Box<dyn RasterQueryProcessor<RasterType = u8>>,
    ) -> Vec<Vec<f64>> {
        let ctx = MockQueryContext::test_default();

        let query_bbox = SpatialPartition2D::new(
            (677770.000, 4764070.000).into(),
            (679870.000, 4762370.000).into(),
        )
        .unwrap();

        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = QueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1072936800000, 1072936800000).unwrap(),
            spatial_resolution: query_spatial_resolution,
        };

        let mut stream = rqp_gt
            .raster_query(qry_rectangle.clone(), &ctx)
            .await
            .unwrap();

        let mut buffer_proc: Vec<Vec<f64>> = Vec::new();

        while let Some(processor) = stream.next().await {
            match processor.unwrap().grid_array {
                GridOrEmpty::Grid(processor) => {
                    let data = &processor.data;
                    println!("{:?}", processor.shape);
                    // TODO: make more generic
                    let data_mapped: Vec<f64> = data.into_iter().map(|elem| *elem as f64).collect();
                    buffer_proc.push(data_mapped);
                }
                _ => {
                    buffer_proc.push(vec![]);
                }
            }
        }

        buffer_proc
    }

    #[tokio::test]
    async fn tangram_rayon_test() {
        // fetch data from geoengine
        let data = prepare_tangram_test_data().await;

        // create thread pool env
        // this is necessary to make the thread pool handling play nice with tangram
        let pool = create_rayon_thread_pool(0);
        let jh = spawn_blocking_with_thread_pool(pool, || {
            // take data and train the tangram tree
            tangram_raster_training(data)
        });

        let predictions = jh.await.unwrap();

        println!("{:?}", predictions);
    }

    async fn prepare_tangram_test_data(
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>> {
        let paths = [
            "raster/landcover/B2_2014-01-01.tif",
            "raster/landcover/B3_2014-01-01.tif",
            "raster/landcover/B4_2014-01-01.tif",
            "raster/landcover/B5_2014-01-01.tif",
            "raster/landcover/B12_2014-01-01.tif",
        ];

        // contains the vectors of all bands.
        // each band consists of multiple rectangles
        // each rectangle is represented by a vec of u8's
        let mut bands: Vec<Vec<Vec<f64>>> = vec![];

        // load each band given by distinct .tif files
        for path in paths.iter() {
            let gcm = get_gdal_config_metadata(path);
            let init_op = initialize_operator(gcm).await;
            let buffer_proc = get_band_data(init_op).await;
            bands.push(buffer_proc);
        }

        // make a single vec for each rectangle
        // TODO: "flatten logic" dürfte noch nicht stimmen, soll heißen die anordnung der pixel ist vermutlich gemischt und nicht in der richtigen reihenfolge
        let mut flat_bands: Vec<Vec<u8>> = vec![];

        for band in bands {
            flat_bands.push(band.into_iter().flatten().map(|elem| elem as u8).collect());
        }

        // make streams from each band
        let stream_vec: Vec<_> = flat_bands
            .into_iter()
            .map(|band| futures::stream::iter(band))
            .collect();

        let svz2 = StreamVectorZip::new(stream_vec);

        // finally collect the data as rows
        let rows: Vec<Vec<u8>> = svz2.collect().await;

        let arr: Vec<f32> = rows
            .iter()
            .flatten()
            .map(|elem| elem.clone() as f32)
            .collect();

        let data = Array2::from_shape_vec((rows.len(), 5), arr).unwrap();

        data
    }

    fn tangram_raster_training(
        data: ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 2]>>,
    ) -> ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<[usize; 1]>> {
        let (x_train_array, x_test_array, y_train_array, y_test_array) =
            get_train_test_split_arrays(data, 4);

        let (x_train, x_test, y_train, y_test) = get_tangram_matrix(
            x_train_array,
            x_test_array,
            y_train_array,
            y_test_array,
            vec!["a".into(), "b".into(), "c".into(), "d".into()],
            "e".into(),
        );

        let train_output = tangram_train_model(ModelType::Numeric, x_train, y_train.clone());

        let arr_size = y_test.len();

        let mut predictions = Array::zeros(arr_size);
        tangram_predict(
            ModelType::Numeric,
            x_test,
            train_output,
            &mut predictions,
            0,
        );

        // println!("{:?}", x_train_array);

        predictions
    }

    #[tokio::test]
    async fn xg_raster_input_test() {
        let paths = [
            "raster/landcover/B2_2014-01-01.tif",
            "raster/landcover/B3_2014-01-01.tif",
            "raster/landcover/B4_2014-01-01.tif",
            "raster/landcover/B5_2014-01-01.tif",
            "raster/landcover/B12_2014-01-01.tif",
        ];

        // contains the vectors of all bands.
        // each band consists of multiple rectangles
        // each rectangle is represented by a vec of u8's
        let mut bands: Vec<Vec<Vec<f64>>> = vec![];

        // load each band given by distinct .tif files
        for path in paths.iter() {
            let gcm = get_gdal_config_metadata(path);
            let init_op = initialize_operator(gcm).await;
            let buffer_proc = get_band_data(init_op).await;
            bands.push(buffer_proc);
        }

        // --------------------------------------
        // just debug stuff

        let b1 = bands.get(0).unwrap();
        let b1_t1 = bands.get(0).unwrap().get(0).unwrap();
        println!("{:?}", b1.len());
        println!("{:?}", b1_t1.len());
        for i in 0..10 {
            println!("{:?}", b1_t1.get(i).unwrap());
        }

        // --------------------------------------

        let stream_vec: Vec<_> = bands
            .into_iter()
            .map(|band| futures::stream::iter(band))
            .collect();

        let svz2 = StreamVectorZip::new(stream_vec);

        let zipped_data: Vec<Vec<Vec<f64>>> = svz2.collect().await;

        // temporary stores for updating xg's data per tile
        let mut booster_vec: Vec<Booster> = Vec::new();
        let mut matrix_vec: Vec<DMatrix> = Vec::new();

        let tile_size = 16;

        let mut reservoir: Vec<&Vec<Vec<f64>>> = Vec::new();
        let capacity = 10;

        let step = Uniform::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let choice: f64 = step.sample(&mut rng);

        let mut v = Vec::new();

        let mut w = (choice.ln() / capacity as f64).exp();
        // iterate over each tile, every time an instance of xgbooster is updated

        let mut next_i = 10;

        for (i, tile) in zipped_data.iter().enumerate() {
            if i < 10 {
                reservoir.push(tile);
            } else if i == next_i {
                // i := i + floor(log(random())/log(1-W)) + 1

                let step = Uniform::new(0.0, 1.0);
                let mut rng = rand::thread_rng();
                let choice: f64 = step.sample(&mut rng);

                let s = (choice.ln() / (1.0 - w).ln()).floor();

                let step = Uniform::new(0.0, 1.0);
                let mut rng = rand::thread_rng();
                let choice: f64 = step.sample(&mut rng);

                w = w * (choice.ln() / capacity as f64).exp();

                next_i = i + s as usize;
                v.push(s);

                dbg!(w);
                dbg!(i);
                dbg!(s);
            }

            // println!("{:?}", &tile);
            // get the band data for each tile
            let band_1 = tile.get(0).unwrap();
            let band_2 = tile.get(1).unwrap();
            let band_3 = tile.get(2).unwrap();
            let band_4 = tile.get(3).unwrap();

            // println!("band.len() {:?}", band_1.len());

            // we need all features (bands) per datapoint (row, coordinate etc)
            let mut tabular_like_data_vec = Vec::new();
            for (a, b, c) in izip!(band_1, band_2, band_3) {
                let row = vec![a.to_owned(), b.to_owned(), c.to_owned()];
                tabular_like_data_vec.extend_from_slice(&row);
            }

            let data_arr_2d =
                Array2::from_shape_vec((i32::pow(tile_size, 2) as usize, 3), tabular_like_data_vec)
                    .unwrap();

            // prepare tecnical metadata for dmatrix
            // xgboost needs the memory information of the data
            let strides_ax_0 = data_arr_2d.strides()[0] as usize;
            let strides_ax_1 = data_arr_2d.strides()[1] as usize;
            let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
            let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

            // get xgboost style matrices
            let mut xg_matrix = DMatrix::from_col_major_f64(
                data_arr_2d.as_slice_memory_order().unwrap(),
                byte_size_ax_0,
                byte_size_ax_1,
                i32::pow(tile_size, 2) as usize,
                3 as usize,
            )
            .unwrap();

            // set labels
            // TODO: make more generic
            let lbls: Vec<f32> = band_4.iter().map(|elem| *elem as f32).collect();
            xg_matrix.set_labels(lbls.as_slice()).unwrap();

            // start actual training
            if booster_vec.len() == 0 {
                println!("generating initial model");

                // in the first iteration, there is no model yet.
                matrix_vec.push(xg_matrix);
                let keys = vec![
                    "validate_parameters",
                    "process_type",
                    "tree_method",
                    "eval_metric",
                    "max_depth",
                ];

                let values = vec!["1", "default", "hist", "rmse", "3"];
                let evals = &[(matrix_vec.get(0).unwrap(), "train")];
                let bst = Booster::my_train(
                    Some(evals),
                    matrix_vec.get(0).unwrap(),
                    keys,
                    values,
                    None, // <- No old model yet
                )
                .unwrap();

                // store the first booster
                booster_vec.push(bst);
            } else {
                println!("updating model");

                // this is a consecutive iteration, so we need the last booster instance
                // to update the model
                let bst = booster_vec.pop().unwrap();

                let keys = vec![
                    "validate_parameters",
                    "process_type",
                    "updater",
                    "refresh_leaf",
                    "eval_metric",
                    "max_depth",
                ];

                let values = vec!["1", "update", "refresh", "true", "rmse", "3"];

                let evals = &[(matrix_vec.get(0).unwrap(), "orig"), (&xg_matrix, "train")];
                let bst_updated = Booster::my_train(
                    Some(evals),
                    &xg_matrix,
                    keys,
                    values,
                    Some(bst), // <- this contains the last model which is now being updated
                )
                .unwrap();

                // store the new booster instance
                booster_vec.push(bst_updated);
            }
        }

        println!("{:?}", v);
        println!("training done");
    }
}
