#[cfg(test)]
mod tests {

    use crate::engine::RasterQueryProcessor;
    use crate::processing::meteosat::{
        new_channel_key, new_offset_key, new_satellite_key, new_slope_key,
    };
    use crate::util::stream_zip::StreamVectorZip;

    use crate::{
        engine::{MockExecutionContext, MockQueryContext, RasterOperator, RasterResultDescriptor},
        source::{
            FileNotFoundHandling, GdalDatasetGeoTransform, GdalDatasetParameters,
            GdalMetaDataRegular, GdalMetadataMapping, GdalSource, GdalSourceParameters,
            GdalSourceTimePlaceholder, TimeReference,
        },
    };

    use futures::StreamExt;

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
    use itertools::{izip, Itertools};
    use ndarray::Array2;
    use rand::distributions::Uniform;
    use rand::prelude::Distribution;
    use xgboost_bindings::{Booster, DMatrix};

    use std::collections::{BTreeMap, HashMap};
    use std::f64;
    use std::mem::{self, size_of};
    use std::path::{Path, PathBuf};

    fn get_gdal_config_metadata(path: &str) -> GdalMetaDataRegular {
        let no_data_value = Some(-339999995214436424907732413799364296704.0);

        let gdal_config_metadata = GdalMetaDataRegular {
            result_descriptor: RasterResultDescriptor {
                data_type: RasterDataType::U8,
                spatial_reference: SpatialReferenceOption::SpatialReference(SpatialReference::new(
                    SpatialReferenceAuthority::Epsg,
                    32618,
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
                width: 208,
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
    async fn xg_raster_input_test() {
        let paths = [
            "B2_2014-01-01.tif",
            "B3_2014-01-01.tif",
            "B4_2014-01-01.tif",
            "LBL_2014-01-01.tif",
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

        let mut w = (choice.ln() / capacity as f64).exp();
        // iterate over each tile, every time an instance of xgbooster is updated

        let mut next_i = 10;

        for (i, tile) in zipped_data.iter().enumerate() {
            if i < 10 {
                reservoir.push(tile);
            } else if i == next_i {
                println!("tile.len(): {:?}", tile.len());
                println!("tile[0].len(): {:?}", tile.get(0).unwrap().len());
                println!("\nTILE ---------------------------------\n");
                println!("current i: {:?}", i);

                let step = Uniform::new(0.0, 9.0);
                let mut rng = rand::thread_rng();
                let swap_elem_idx: f64 = step.sample(&mut rng);

                reservoir.push(tile);
                reservoir.swap_remove(swap_elem_idx as usize);
                println!("reservoir len: {:?}", reservoir.len());

                let step = Uniform::new(0.0, 1.0);
                let mut rng = rand::thread_rng();
                let choice: f64 = step.sample(&mut rng);

                let s = (choice.ln() / (1.0 - w).ln()).floor();

                let step = Uniform::new(0.0, 1.0);
                let mut rng = rand::thread_rng();
                let choice: f64 = step.sample(&mut rng);

                w = w * (choice.ln() / capacity as f64).exp();

                next_i = i + 1 + s as usize;

                println!("next_i: {:?}", next_i);

                let mut band_1 = Vec::new();
                let mut band_2 = Vec::new();
                let mut band_3 = Vec::new();
                let mut band_4 = Vec::new();

                // go over all tiles in the reservoir
                // and make a collective band from all tiles
                for tile in reservoir.iter() {
                    for elem in tile.get(0).unwrap().iter() {
                        band_1.push(elem);
                    }
                    for elem in tile.get(1).unwrap().iter() {
                        band_2.push(elem);
                    }
                    for elem in tile.get(2).unwrap().iter() {
                        band_3.push(elem);
                    }
                    for elem in tile.get(3).unwrap().iter() {
                        band_4.push(elem);
                    }
                }

                // we need all features (bands) per datapoint (row, coordinate etc)
                let mut tabular_like_data_vec = Vec::new();
                for (a, b, c) in izip!(band_1, band_2, band_3) {
                    let row = vec![a.to_owned(), b.to_owned(), c.to_owned()];
                    tabular_like_data_vec.extend_from_slice(&row);
                }

                let data_arr_2d = Array2::from_shape_vec(
                    (i32::pow(tile_size, 2) as usize * 10, 3),
                    tabular_like_data_vec,
                )
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
                let lbls: Vec<f32> = band_4.iter().map(|elem| **elem as f32).collect();
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
        }

        println!("training done.");
    }

    /// This function calculates the maximum possible reservoir size for the given parameters.
    /// # Arguments
    /// max_mem_cap: How big the maximum memory capacity can be (in given unit).
    /// unit: What the unit of the mem cap is. -> Megabytes etc.
    /// n_bands: How many bands are read.
    /// type_size: The size of the elements in the bands.
    /// types: A vector of type sizes. Contains information if bands are of different types. TODO
    fn calculate_reservoir_size(
        max_mem_cap: usize,
        unit: &str,
        n_bands: usize,
        type_size: usize,
        types: Option<Vec<usize>>,
    ) -> usize {
        // TODO: @param types implementieren
        let factor = match unit.to_lowercase().as_str() {
            "kb" => 1024,
            "mb" => 1024 * 1024,
            "gb" => 1024 * 1024 * 1024,
            _ => 1,
        };

        let n_bytes_per_band = (max_mem_cap * factor) / n_bands;
        let n_elements_per_band = n_bytes_per_band / type_size;

        println!(
            "possible reservoir size is {:?} elements (per band)",
            n_elements_per_band
        );

        n_elements_per_band
    }

    #[test]
    fn mem_size_test() {
        calculate_reservoir_size(1, "kb", 4, mem::size_of::<f64>(), None);
    }

    #[tokio::test]
    async fn xg_reservoir_test() {
        let paths = [
            "raster/landcover/B2_2014-01-01.tif",
            "raster/landcover/B3_2014-01-01.tif",
            "raster/landcover/B4_2014-01-01.tif",
            "raster/landcover/Class_ID_2014-01-01.tif",
        ];

        let capacity = calculate_reservoir_size(1, "kb", 4, mem::size_of::<f64>(), None);

        let mut booster_vec: Vec<Booster> = Vec::new();
        let mut matrix_vec: Vec<DMatrix> = Vec::new();

        let mut vec_of_reservoir_indices = vec![];

        for _ in 0..500 {
            let mut reservoir_indices = Vec::new();
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

            let stream_vec: Vec<_> = bands
                .into_iter()
                .map(|band| futures::stream::iter(band))
                .collect();

            let svz2 = StreamVectorZip::new(stream_vec);

            let zipped_data: Vec<Vec<Vec<f64>>> = svz2.collect().await;

            let tile_size = 16;

            let step = Uniform::new(0.0, 1.0);
            let mut rng = rand::thread_rng();
            let choice: f64 = step.sample(&mut rng);

            let mut w = (choice.ln() / capacity as f64).exp();
            // iterate over each tile, every time an instance of xgbooster is updated

            let mut i = 0;

            let mut reservoir_b1: Vec<_> = Vec::new();
            let mut reservoir_b2: Vec<_> = Vec::new();
            let mut reservoir_target: Vec<_> = Vec::new();

            // go over the complete datastream and generate a reservoir from it
            for (tile_counter, tile) in zipped_data.iter().enumerate() {
                // check if next element is in current tile or we can skip this tile
                if i >= (tile_counter + 1) * (tile_size * tile_size) {
                    continue;
                }

                // initial fill of the reservoir
                if i < capacity {
                    let band_1 = tile.get(0).unwrap();
                    let band_2 = tile.get(1).unwrap();
                    let band_target = tile.get(3).unwrap();
                    for (b1, b2, target) in izip!(band_1, band_2, band_target) {
                        reservoir_b1.push(b1);
                        reservoir_b2.push(b2);
                        reservoir_target.push(target);

                        reservoir_indices.push(i);
                        i = i + 1;
                    }
                }
                // consecutive fill of the reservoir with random elements
                else {
                    while i < (tile_counter + 1) * (tile_size * tile_size) {
                        let step = Uniform::new(0, 1024);
                        let mut rng = rand::thread_rng();
                        let idx_swap_elem = step.sample(&mut rng);

                        let idx_this_tile = i - (tile_counter * (tile_size * tile_size));

                        // change element
                        let a = tile.get(0).unwrap().get(idx_this_tile).unwrap();
                        let b = tile.get(1).unwrap().get(idx_this_tile).unwrap();
                        let c = tile.get(3).unwrap().get(idx_this_tile).unwrap();
                        reservoir_b1.push(a);
                        reservoir_b2.push(b);
                        reservoir_target.push(c);

                        reservoir_b1.swap_remove(idx_swap_elem);
                        reservoir_b2.swap_remove(idx_swap_elem);
                        reservoir_target.swap_remove(idx_swap_elem);

                        // save indices to check distribution of reservoir elements
                        reservoir_indices.push(i);
                        reservoir_indices.swap_remove(idx_swap_elem);

                        let step = Uniform::new(0.0, 1.0);
                        let mut rng = rand::thread_rng();
                        let choice: f64 = step.sample(&mut rng);

                        let s = (choice.ln() / (1.0 - w).ln()).floor();

                        let step = Uniform::new(0.0, 1.0);
                        let mut rng = rand::thread_rng();
                        let choice: f64 = step.sample(&mut rng);

                        w = w * (choice.ln() / capacity as f64).exp();

                        i = i + 1 + s as usize;
                    }
                }
            }

            // train with the generated reservoir
            let mut tabular_like_data_vec = Vec::new();
            for (a, b) in izip!(reservoir_b1, reservoir_b2) {
                let row = vec![a.to_owned(), b.to_owned()];
                tabular_like_data_vec.extend_from_slice(&row);
            }

            let data_arr_2d = Array2::from_shape_vec((capacity, 2), tabular_like_data_vec).unwrap();

            let strides_ax_0 = data_arr_2d.strides()[0] as usize;
            let strides_ax_1 = data_arr_2d.strides()[1] as usize;
            let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
            let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

            // get xgboost style matrices
            let mut xg_matrix = DMatrix::from_col_major_f64(
                data_arr_2d.as_slice_memory_order().unwrap(),
                byte_size_ax_0,
                byte_size_ax_1,
                capacity,
                2 as usize,
            )
            .unwrap();

            // set labels
            // TODO: make more generic
            let lbls: Vec<f32> = reservoir_target.iter().map(|elem| **elem as f32).collect();
            xg_matrix.set_labels(lbls.as_slice()).unwrap();

            // initial training round
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
            }
            // update training rounds
            else {
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
            vec_of_reservoir_indices.push(reservoir_indices);
        }

        // collect information for dbg
        let all_chosen_elements = vec_of_reservoir_indices.iter().flatten().collect_vec();

        // serialize hashmap to csv
        let mut wtr = csv::Writer::from_path(Path::new("test.csv")).unwrap();

        wtr.write_record(&["index"]).unwrap();

        for elem in all_chosen_elements.iter() {
            let val = format!("{elem}");
            wtr.write_record(&[val]).unwrap();
        }

        wtr.flush().unwrap();

        println!("training done");
    }

    #[tokio::test]
    async fn workflow() {
        // define data to be used
        let paths = [
            "raster/landcover/B2_2014-01-01.tif",
            "raster/landcover/B3_2014-01-01.tif",
            "raster/landcover/B4_2014-01-01.tif",
            "raster/landcover/Class_ID_2014-01-01.tif",
        ];

        // define reservoir size
        let capacity = calculate_reservoir_size(1, "kb", 4, mem::size_of::<f64>(), None);

        // how many rounds should be trained?
        let mut booster_vec: Vec<Booster> = Vec::new();
        let mut matrix_vec: Vec<DMatrix> = Vec::new();
        let training_rounds = 5;

        let mut bands: Vec<Vec<Vec<f64>>> = vec![];

        // load each band given by distinct .tif files
        for path in paths.iter() {
            let gcm = get_gdal_config_metadata(path);
            let init_op = initialize_operator(gcm).await;
            let buffer_proc = get_band_data(init_op).await;
            bands.push(buffer_proc);
        }

        let stream_vec: Vec<_> = bands
            .into_iter()
            .map(|band| futures::stream::iter(band))
            .collect();

        let svz2 = StreamVectorZip::new(stream_vec);

        let zipped_data: Vec<Vec<Vec<f64>>> = svz2.collect().await;

        let tile_size = 16;

        for _ in 0..training_rounds {
            // generate a reservoir
            let res = generate_reservoir(zipped_data.clone(), tile_size, capacity).await;

            // make xg compatible, trainable datastructure
            let xg_matrix = make_xg_data(vec![res.0, res.1, res.2], capacity);

            train_model(&mut booster_vec, &mut matrix_vec, xg_matrix);

            // initial training round
        }
    }

    fn train_model(
        booster_vec: &mut Vec<Booster>,
        matrix_vec: &mut Vec<DMatrix>,
        xg_matrix: DMatrix,
    ) {
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
        }
        // update training rounds
        else {
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

    fn make_xg_data(res: Vec<Vec<f64>>, capacity: usize) -> DMatrix {
        let mut tabular_like_data_vec = Vec::new();
        for (a, b) in izip!(res.get(0).unwrap(), res.get(1).unwrap()) {
            let row = vec![a.to_owned(), b.to_owned()];
            tabular_like_data_vec.extend_from_slice(&row);
        }

        let data_arr_2d = Array2::from_shape_vec((capacity, 2), tabular_like_data_vec).unwrap();

        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let mut xg_matrix = DMatrix::from_col_major_f64(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            capacity,
            2 as usize,
        )
        .unwrap();

        // set labels
        // TODO: make more generic
        let lbls: Vec<f32> = res
            .get(2)
            .unwrap()
            .iter()
            .map(|elem| *elem as f32)
            .collect();
        xg_matrix.set_labels(lbls.as_slice()).unwrap();
        xg_matrix
    }

    async fn generate_reservoir<'a>(
        zipped_data: Vec<Vec<Vec<f64>>>,
        tile_size: usize,
        capacity: usize,
    ) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut i = 0;
        let mut reservoir_b1: Vec<f64> = Vec::new();
        let mut reservoir_b2: Vec<f64> = Vec::new();
        let mut reservoir_target: Vec<f64> = Vec::new();

        let step = Uniform::new(0.0, 1.0);
        let mut rng = rand::thread_rng();
        let choice: f64 = step.sample(&mut rng);

        let mut w = (choice.ln() / capacity as f64).exp();

        for (tile_counter, tile) in zipped_data.iter().enumerate() {
            // check if next element is in current tile or we can skip this tile
            if elem_not_in_this_tile(i, tile_counter, tile_size) == true {
                continue;
            }
            if i >= (tile_counter + 1) * (tile_size * tile_size) {
                continue;
            }

            let b1 = tile.get(0).unwrap();
            let b2 = tile.get(1).unwrap();
            let target = tile.get(3).unwrap();

            // initial fill of the reservoir
            if i < capacity {
                while i < capacity {
                    let elem_b1 = b1.get(i).unwrap();
                    let elem_b2 = b2.get(i).unwrap();
                    let elem_target = target.get(i).unwrap();
                    reservoir_b1.push(elem_b1.to_owned());
                    reservoir_b2.push(elem_b2.to_owned());
                    reservoir_target.push(elem_target.to_owned());
                    i = i + 1;
                }
            }
            // consecutive fill of the reservoir with random elements
            else {
                while i < (tile_counter + 1) * (tile_size * tile_size) {
                    let step = Uniform::new(0, capacity);
                    let mut rng = rand::thread_rng();
                    let idx_swap_elem = step.sample(&mut rng);

                    let idx_this_tile = i - (tile_counter * (tile_size * tile_size));

                    // change element
                    let a = tile.get(0).unwrap().get(idx_this_tile).unwrap();
                    let b = tile.get(1).unwrap().get(idx_this_tile).unwrap();
                    let c = tile.get(3).unwrap().get(idx_this_tile).unwrap();
                    reservoir_b1.push(a.to_owned());
                    reservoir_b2.push(b.to_owned());
                    reservoir_target.push(c.to_owned());

                    let l = reservoir_b2.len();

                    reservoir_b1.swap_remove(idx_swap_elem);
                    reservoir_b2.swap_remove(idx_swap_elem);
                    reservoir_target.swap_remove(idx_swap_elem);

                    // save indices to check distribution of reservoir elements

                    let step = Uniform::new(0.0, 1.0);
                    let mut rng = rand::thread_rng();
                    let choice: f64 = step.sample(&mut rng);

                    let s = (choice.ln() / (1.0 - w).ln()).floor();

                    let step = Uniform::new(0.0, 1.0);
                    let mut rng = rand::thread_rng();
                    let choice: f64 = step.sample(&mut rng);

                    w = w * (choice.ln() / capacity as f64).exp();

                    i = i + 1 + s as usize;
                }
            }
        }

        (reservoir_b1, reservoir_b2, reservoir_target)
    }

    fn elem_not_in_this_tile(i: usize, tile_counter: usize, tile_size: usize) -> bool{
        i >= (tile_counter + 1) * (tile_size * tile_size)
    }
}
