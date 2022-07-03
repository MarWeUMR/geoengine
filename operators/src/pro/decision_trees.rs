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
    use ndarray::{arr2, Array2};
    use rand::distributions::Uniform;
    use rand::prelude::Distribution;
    use xgboost_bindings::{Booster, DMatrix};

    use std::cmp::Ordering;
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
                    origin_coordinate: (474112.0, 5646336.0).into(),
                    x_pixel_size: 10.0,
                    y_pixel_size: -10.0,
                },
                width: 4864,
                height: 3431,
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
        tile_size: usize,
    ) -> Box<dyn RasterQueryProcessor<RasterType = u8>> {
        let tiling_specification =
            TilingSpecification::new(Coordinate2D::default(), [tile_size, tile_size].into());

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
            (474112.000, 5646336.000).into(),
            (522752.000, 5612026.000).into(),
        )
        .unwrap();
        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = QueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1590969600000, 1590969600000).unwrap(),
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

    // #[tokio::test]
    // async fn xg_raster_input_test() {
    //     let paths = [
    //         "B2_2014-01-01.tif",
    //         "B3_2014-01-01.tif",
    //         "B4_2014-01-01.tif",
    //         "LBL_2014-01-01.tif",
    //     ];
    //
    //     // contains the vectors of all bands.
    //     // each band consists of multiple rectangles
    //     // each rectangle is represented by a vec of u8's
    //     let mut bands: Vec<Vec<Vec<f64>>> = vec![];
    //
    //     // load each band given by distinct .tif files
    //     for path in paths.iter() {
    //         let gcm = get_gdal_config_metadata(path);
    //         let init_op = initialize_operator(gcm, tile_size).await;
    //         let buffer_proc = get_band_data(init_op).await;
    //         bands.push(buffer_proc);
    //     }
    //
    //     // --------------------------------------
    //     // just debug stuff
    //
    //     let b1 = bands.get(0).unwrap();
    //     let b1_t1 = bands.get(0).unwrap().get(0).unwrap();
    //     println!("{:?}", b1.len());
    //     println!("{:?}", b1_t1.len());
    //     for i in 0..10 {
    //         println!("{:?}", b1_t1.get(i).unwrap());
    //     }
    //
    //     // --------------------------------------
    //
    //     let stream_vec: Vec<_> = bands
    //         .into_iter()
    //         .map(|band| futures::stream::iter(band))
    //         .collect();
    //
    //     let svz2 = StreamVectorZip::new(stream_vec);
    //
    //     let zipped_data: Vec<Vec<Vec<f64>>> = svz2.collect().await;
    //
    //     // temporary stores for updating xg's data per tile
    //     let mut booster_vec: Vec<Booster> = Vec::new();
    //     let mut matrix_vec: Vec<DMatrix> = Vec::new();
    //
    //     let tile_size = 16;
    //
    //     let mut reservoir: Vec<&Vec<Vec<f64>>> = Vec::new();
    //     let capacity = 10;
    //
    //     let step = Uniform::new(0.0, 1.0);
    //     let mut rng = rand::thread_rng();
    //     let choice: f64 = step.sample(&mut rng);
    //
    //     let mut w = (choice.ln() / capacity as f64).exp();
    //     // iterate over each tile, every time an instance of xgbooster is updated
    //
    //     let mut next_i = 10;
    //
    //     for (i, tile) in zipped_data.iter().enumerate() {
    //         if i < 10 {
    //             reservoir.push(tile);
    //         } else if i == next_i {
    //             println!("tile.len(): {:?}", tile.len());
    //             println!("tile[0].len(): {:?}", tile.get(0).unwrap().len());
    //             println!("\nTILE ---------------------------------\n");
    //             println!("current i: {:?}", i);
    //
    //             let step = Uniform::new(0.0, 9.0);
    //             let mut rng = rand::thread_rng();
    //             let swap_elem_idx: f64 = step.sample(&mut rng);
    //
    //             reservoir.push(tile);
    //             reservoir.swap_remove(swap_elem_idx as usize);
    //             println!("reservoir len: {:?}", reservoir.len());
    //
    //             let step = Uniform::new(0.0, 1.0);
    //             let mut rng = rand::thread_rng();
    //             let choice: f64 = step.sample(&mut rng);
    //
    //             let s = (choice.ln() / (1.0 - w).ln()).floor();
    //
    //             let step = Uniform::new(0.0, 1.0);
    //             let mut rng = rand::thread_rng();
    //             let choice: f64 = step.sample(&mut rng);
    //
    //             w = w * (choice.ln() / capacity as f64).exp();
    //
    //             next_i = i + 1 + s as usize;
    //
    //             println!("next_i: {:?}", next_i);
    //
    //             let mut band_1 = Vec::new();
    //             let mut band_2 = Vec::new();
    //             let mut band_3 = Vec::new();
    //             let mut band_4 = Vec::new();
    //
    //             // go over all tiles in the reservoir
    //             // and make a collective band from all tiles
    //             for tile in reservoir.iter() {
    //                 for elem in tile.get(0).unwrap().iter() {
    //                     band_1.push(elem);
    //                 }
    //                 for elem in tile.get(1).unwrap().iter() {
    //                     band_2.push(elem);
    //                 }
    //                 for elem in tile.get(2).unwrap().iter() {
    //                     band_3.push(elem);
    //                 }
    //                 for elem in tile.get(3).unwrap().iter() {
    //                     band_4.push(elem);
    //                 }
    //             }
    //
    //             // we need all features (bands) per datapoint (row, coordinate etc)
    //             let mut tabular_like_data_vec = Vec::new();
    //             for (a, b, c) in izip!(band_1, band_2, band_3) {
    //                 let row = vec![a.to_owned(), b.to_owned(), c.to_owned()];
    //                 tabular_like_data_vec.extend_from_slice(&row);
    //             }
    //
    //             let data_arr_2d = Array2::from_shape_vec(
    //                 (i32::pow(tile_size, 2) as usize * 10, 3),
    //                 tabular_like_data_vec,
    //             )
    //             .unwrap();
    //
    //             // prepare tecnical metadata for dmatrix
    //             // xgboost needs the memory information of the data
    //             let strides_ax_0 = data_arr_2d.strides()[0] as usize;
    //             let strides_ax_1 = data_arr_2d.strides()[1] as usize;
    //             let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
    //             let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;
    //
    //             // get xgboost style matrices
    //             let mut xg_matrix = DMatrix::from_col_major_f64(
    //                 data_arr_2d.as_slice_memory_order().unwrap(),
    //                 byte_size_ax_0,
    //                 byte_size_ax_1,
    //                 i32::pow(tile_size, 2) as usize,
    //                 3 as usize,
    //             )
    //             .unwrap();
    //
    //             // set labels
    //             // TODO: make more generic
    //             let lbls: Vec<f32> = band_4.iter().map(|elem| **elem as f32).collect();
    //             xg_matrix.set_labels(lbls.as_slice()).unwrap();
    //
    //             // start actual training
    //             if booster_vec.len() == 0 {
    //                 println!("generating initial model");
    //
    //                 // in the first iteration, there is no model yet.
    //                 matrix_vec.push(xg_matrix);
    //                 let keys = vec![
    //                     "validate_parameters",
    //                     "process_type",
    //                     "tree_method",
    //                     "eval_metric",
    //                     "max_depth",
    //                 ];
    //
    //                 let values = vec!["1", "default", "hist", "rmse", "3"];
    //                 let evals = &[(matrix_vec.get(0).unwrap(), "train")];
    //                 let bst = Booster::my_train(
    //                     Some(evals),
    //                     matrix_vec.get(0).unwrap(),
    //                     keys,
    //                     values,
    //                     None, // <- No old model yet
    //                 )
    //                 .unwrap();
    //
    //                 // store the first booster
    //                 booster_vec.push(bst);
    //             } else {
    //                 println!("updating model");
    //
    //                 // this is a consecutive iteration, so we need the last booster instance
    //                 // to update the model
    //                 let bst = booster_vec.pop().unwrap();
    //
    //                 let keys = vec![
    //                     "validate_parameters",
    //                     "process_type",
    //                     "updater",
    //                     "refresh_leaf",
    //                     "eval_metric",
    //                     "max_depth",
    //                 ];
    //
    //                 let values = vec!["1", "update", "refresh", "true", "rmse", "3"];
    //
    //                 let evals = &[(matrix_vec.get(0).unwrap(), "orig"), (&xg_matrix, "train")];
    //                 let bst_updated = Booster::my_train(
    //                     Some(evals),
    //                     &xg_matrix,
    //                     keys,
    //                     values,
    //                     Some(bst), // <- this contains the last model which is now being updated
    //                 )
    //                 .unwrap();
    //
    //                 // store the new booster instance
    //                 booster_vec.push(bst_updated);
    //             }
    //         }
    //     }
    //
    //     println!("training done.");
    // }

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

    // #[tokio::test]
    // async fn xg_reservoir_test() {
    //     let paths = [
    //         "raster/landcover/B2_2014-01-01.tif",
    //         "raster/landcover/B3_2014-01-01.tif",
    //         "raster/landcover/B4_2014-01-01.tif",
    //         "raster/landcover/Class_ID_2014-01-01.tif",
    //     ];
    //
    //     let capacity = calculate_reservoir_size(1, "kb", 4, mem::size_of::<f64>(), None);
    //
    //     let mut booster_vec: Vec<Booster> = Vec::new();
    //     let mut matrix_vec: Vec<DMatrix> = Vec::new();
    //
    //     let tile_size = 16;
    //     let mut vec_of_reservoir_indices = vec![];
    //
    //     for _ in 0..500 {
    //         let mut reservoir_indices = Vec::new();
    //         // contains the vectors of all bands.
    //         // each band consists of multiple rectangles
    //         // each rectangle is represented by a vec of u8's
    //         let mut bands: Vec<Vec<Vec<f64>>> = vec![];
    //
    //         // load each band given by distinct .tif files
    //         for path in paths.iter() {
    //             let gcm = get_gdal_config_metadata(path);
    //             let init_op = initialize_operator(gcm, tile_size).await;
    //             let buffer_proc = get_band_data(init_op).await;
    //             bands.push(buffer_proc);
    //         }
    //
    //         let stream_vec: Vec<_> = bands
    //             .into_iter()
    //             .map(|band| futures::stream::iter(band))
    //             .collect();
    //
    //         let svz2 = StreamVectorZip::new(stream_vec);
    //
    //         let zipped_data: Vec<Vec<Vec<f64>>> = svz2.collect().await;
    //
    //         let step = Uniform::new(0.0, 1.0);
    //         let mut rng = rand::thread_rng();
    //         let choice: f64 = step.sample(&mut rng);
    //
    //         let mut w = (choice.ln() / capacity as f64).exp();
    //         // iterate over each tile, every time an instance of xgbooster is updated
    //
    //         let mut i = 0;
    //
    //         let mut reservoir_b1: Vec<_> = Vec::new();
    //         let mut reservoir_b2: Vec<_> = Vec::new();
    //         let mut reservoir_target: Vec<_> = Vec::new();
    //
    //         // go over the complete datastream and generate a reservoir from it
    //         for (tile_counter, tile) in zipped_data.iter().enumerate() {
    //             // check if next element is in current tile or we can skip this tile
    //             if i >= (tile_counter + 1) * (tile_size * tile_size) {
    //                 continue;
    //             }
    //
    //             // initial fill of the reservoir
    //             if i < capacity {
    //                 let band_1 = tile.get(0).unwrap();
    //                 let band_2 = tile.get(1).unwrap();
    //                 let band_target = tile.get(3).unwrap();
    //                 for (b1, b2, target) in izip!(band_1, band_2, band_target) {
    //                     reservoir_b1.push(b1);
    //                     reservoir_b2.push(b2);
    //                     reservoir_target.push(target);
    //
    //                     reservoir_indices.push(i);
    //                     i = i + 1;
    //                 }
    //             }
    //             // consecutive fill of the reservoir with random elements
    //             else {
    //                 while i < (tile_counter + 1) * (tile_size * tile_size) {
    //                     let step = Uniform::new(0, 1024);
    //                     let mut rng = rand::thread_rng();
    //                     let idx_swap_elem = step.sample(&mut rng);
    //
    //                     let idx_this_tile = i - (tile_counter * (tile_size * tile_size));
    //
    //                     // change element
    //                     let a = tile.get(0).unwrap().get(idx_this_tile).unwrap();
    //                     let b = tile.get(1).unwrap().get(idx_this_tile).unwrap();
    //                     let c = tile.get(3).unwrap().get(idx_this_tile).unwrap();
    //                     reservoir_b1.push(a);
    //                     reservoir_b2.push(b);
    //                     reservoir_target.push(c);
    //
    //                     reservoir_b1.swap_remove(idx_swap_elem);
    //                     reservoir_b2.swap_remove(idx_swap_elem);
    //                     reservoir_target.swap_remove(idx_swap_elem);
    //
    //                     // save indices to check distribution of reservoir elements
    //                     reservoir_indices.push(i);
    //                     reservoir_indices.swap_remove(idx_swap_elem);
    //
    //                     let step = Uniform::new(0.0, 1.0);
    //                     let mut rng = rand::thread_rng();
    //                     let choice: f64 = step.sample(&mut rng);
    //
    //                     let s = (choice.ln() / (1.0 - w).ln()).floor();
    //
    //                     let step = Uniform::new(0.0, 1.0);
    //                     let mut rng = rand::thread_rng();
    //                     let choice: f64 = step.sample(&mut rng);
    //
    //                     w = w * (choice.ln() / capacity as f64).exp();
    //
    //                     i = i + 1 + s as usize;
    //                 }
    //             }
    //         }
    //
    //         // train with the generated reservoir
    //         let mut tabular_like_data_vec = Vec::new();
    //         for (a, b) in izip!(reservoir_b1, reservoir_b2) {
    //             let row = vec![a.to_owned(), b.to_owned()];
    //             tabular_like_data_vec.extend_from_slice(&row);
    //         }
    //
    //         let data_arr_2d = Array2::from_shape_vec((capacity, 2), tabular_like_data_vec).unwrap();
    //
    //         let strides_ax_0 = data_arr_2d.strides()[0] as usize;
    //         let strides_ax_1 = data_arr_2d.strides()[1] as usize;
    //         let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
    //         let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;
    //
    //         // get xgboost style matrices
    //         let mut xg_matrix = DMatrix::from_col_major_f64(
    //             data_arr_2d.as_slice_memory_order().unwrap(),
    //             byte_size_ax_0,
    //             byte_size_ax_1,
    //             capacity,
    //             2 as usize,
    //         )
    //         .unwrap();
    //
    //         // set labels
    //         // TODO: make more generic
    //         let lbls: Vec<f32> = reservoir_target.iter().map(|elem| **elem as f32).collect();
    //         xg_matrix.set_labels(lbls.as_slice()).unwrap();
    //
    //         // initial training round
    //         if booster_vec.len() == 0 {
    //             println!("generating initial model");
    //
    //             // in the first iteration, there is no model yet.
    //             matrix_vec.push(xg_matrix);
    //             let keys = vec![
    //                 "validate_parameters",
    //                 "process_type",
    //                 "tree_method",
    //                 "eval_metric",
    //                 "max_depth",
    //             ];
    //
    //             let values = vec!["1", "default", "hist", "rmse", "3"];
    //             let evals = &[(matrix_vec.get(0).unwrap(), "train")];
    //             let bst = Booster::train(
    //                 Some(evals),
    //                 matrix_vec.get(0).unwrap(),
    //                 keys,
    //                 values,
    //                 None, // <- No old model yet
    //             )
    //             .unwrap();
    //
    //             // store the first booster
    //             booster_vec.push(bst);
    //         }
    //         // update training rounds
    //         else {
    //             println!("updating model");
    //
    //             // this is a consecutive iteration, so we need the last booster instance
    //             // to update the model
    //             let bst = booster_vec.pop().unwrap();
    //
    //             let keys = vec![
    //                 "validate_parameters",
    //                 "process_type",
    //                 "updater",
    //                 "refresh_leaf",
    //                 "eval_metric",
    //                 "max_depth",
    //             ];
    //
    //             let values = vec!["1", "update", "refresh", "true", "rmse", "3"];
    //
    //             let evals = &[(matrix_vec.get(0).unwrap(), "orig"), (&xg_matrix, "train")];
    //             let bst_updated = Booster::train(
    //                 Some(evals),
    //                 &xg_matrix,
    //                 keys,
    //                 values,
    //                 Some(bst), // <- this contains the last model which is now being updated
    //             )
    //             .unwrap();
    //
    //             // store the new booster instance
    //             booster_vec.push(bst_updated);
    //         }
    //         vec_of_reservoir_indices.push(reservoir_indices);
    //     }
    //
    //     // collect information for dbg
    //     let all_chosen_elements = vec_of_reservoir_indices.iter().flatten().collect_vec();
    //
    //     // serialize hashmap to csv
    //     let mut wtr = csv::Writer::from_path(Path::new("test.csv")).unwrap();
    //
    //     wtr.write_record(&["index"]).unwrap();
    //
    //     for elem in all_chosen_elements.iter() {
    //         let val = format!("{elem}");
    //         wtr.write_record(&[val]).unwrap();
    //     }
    //
    //     wtr.flush().unwrap();
    //
    //     println!("training done");
    // }

    #[tokio::test]
    async fn workflow() {
        // define data to be used
        // let paths = [
        //     "raster/landcover/B2_2014-01-01.tif",
        //     "raster/landcover/B3_2014-01-01.tif",
        //     "raster/landcover/B4_2014-01-01.tif",
        //     "raster/landcover/Class_ID_2014-01-01.tif",
        // ];

        let paths = [
            "s2_10m_de_marburg/b02.tiff",
            "s2_10m_de_marburg/b03.tiff",
            "s2_10m_de_marburg/b04.tiff",
            "s2_10m_de_marburg/b08.tiff",
            "s2_10m_de_marburg/target.tiff",
        ];

        // define reservoir size
        let capacity = calculate_reservoir_size(10, "mb", paths.len(), mem::size_of::<f64>(), None);

        // how many rounds should be trained?
        let training_rounds = 1;

        // needs to be a power of 2
        // TODO: remove this parameter?
        let tile_size = 512;

        // setup data/model cache
        let mut booster_vec: Vec<Booster> = Vec::new();
        let mut matrix_vec: Vec<DMatrix> = Vec::new();

        // do geoengine magic
        let zipped_data: Vec<Vec<Vec<f64>>> = zip_bands_to_tiles(&paths, tile_size).await;

        let mut forward_map = HashMap::new();
        let mut true_distribution_map = HashMap::new();

        for _ in 0..training_rounds {
            println!("training with a reservoir size of {:?}", capacity);

            // generate and fill a reservoir
            let reservoirs = generate_reservoir(&zipped_data, tile_size, capacity).await;

            // make xg compatible, trainable datastructure
            let xg_matrix = make_xg_data(
                reservoirs,
                capacity,
                &mut forward_map,
                &mut true_distribution_map,
            )
            .await;

            // start the training process
            // TODO: num_rounds implementieren
            train_model(&mut booster_vec, &mut matrix_vec, xg_matrix);
        }

        // predict data
        let mut predictions = predict(booster_vec.pop().unwrap()).await.unwrap();

        // remap predictions to original data
        // let mut remapped_predictions = Vec::new();

        // create the backwards hashmap
        let mut backward_map = HashMap::new();
        for (key, value) in forward_map.iter() {
            backward_map.insert(value, key);
        }

        // now count the predicted values
        let mut result = HashMap::new();
        for elem in predictions.iter() {
            let x = backward_map.get(&(*elem as i32)).unwrap();
            *result.entry(format!("{x}")).or_insert(0) += 1;
        }

        let count_b: BTreeMap<&i32, &String> = true_distribution_map.iter().map(|(k, v)| (v, k)).collect();

        println!("true distribution: {:?}", count_b);
        println!("predicted distribution: {:?}", result);
        println!("forward map: {:?}", forward_map);
        println!("backward map: {:?}", backward_map);
        println!("done");
    }

    /// This function takes a slice of paths to 'band_i.tif' files and turns them into a vector of zipped, tiled data.
    /// Each band is tiled in the beginning. The elements per tile are then zipped together, such that a tabular style
    /// result is returned.
    /// The result contains all tiles of all bands of the provided data.
    /// The structure looks like this:
    /// Zipped_data[
    ///     Tile_1[
    ///        Band_1[elem1,...,elem_tilesize^2],
    ///        ...,
    ///        Band_n[elem1,...,elem_tilesize^2]
    ///           ],
    ///     ...,
    ///    Tile_n[
    ///        Band_1[elem1,...,elem_tilesize^2],
    ///        ...,
    ///        Band_n[elem1,...,elem_tilesize^2]
    ///           ]
    ///   ]

    async fn zip_bands_to_tiles(paths: &[&str], tile_size: usize) -> Vec<Vec<Vec<f64>>> {
        let mut bands: Vec<Vec<Vec<f64>>> = vec![];

        // load each band given by distinct .tif files
        for path in paths.iter() {
            let gcm = get_gdal_config_metadata(path);
            let init_op = initialize_operator(gcm, tile_size).await;
            let buffer_proc = get_band_data(init_op).await;
            bands.push(buffer_proc);
        }

        let streamed_bands: Vec<_> = bands
            .into_iter()
            .map(|band| futures::stream::iter(band))
            .collect();

        let zipped_bands = StreamVectorZip::new(streamed_bands);

        let tiles_of_zipped_bands: Vec<Vec<Vec<f64>>> = zipped_bands.collect().await;

        tiles_of_zipped_bands
    }

    // TODO: how to integrate the configs?
    fn train_model(
        booster_vec: &mut Vec<Booster>,
        matrix_vec: &mut Vec<DMatrix>,
        xg_matrix: DMatrix,
    ) {
        if booster_vec.len() == 0 {
            println!("generating initial model");

            // in the first iteration, there is no model yet.
            matrix_vec.push(xg_matrix);

            let mut initial_training_config: HashMap<&str, &str> = HashMap::new();

            initial_training_config.insert("validate_parameters", "1");
            initial_training_config.insert("process_type", "default");
            initial_training_config.insert("tree_method", "hist");
            initial_training_config.insert("max_depth", "6");
            initial_training_config.insert("objective", "multi:softmax");
            initial_training_config.insert("num_class", "7");
            initial_training_config.insert("eta", "0.5");

            let evals = &[(matrix_vec.get(0).unwrap(), "train")];
            let bst = Booster::train(
                Some(evals),
                matrix_vec.get(0).unwrap(),
                initial_training_config,
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

            let mut update_training_config: HashMap<&str, &str> = HashMap::new();

            update_training_config.insert("validate_parameters", "1");
            update_training_config.insert("process_type", "update");
            update_training_config.insert("updater", "refresh");
            update_training_config.insert("refresh_leaf", "true");
            update_training_config.insert("objective", "multi:softmax");
            update_training_config.insert("num_class", "7");
            update_training_config.insert("max_depth", "6");

            let evals = &[(matrix_vec.get(0).unwrap(), "orig"), (&xg_matrix, "train")];
            let bst_updated = Booster::train(
                Some(evals),
                &xg_matrix,
                update_training_config,
                Some(bst), // <- this contains the last model which is now being updated
            )
            .unwrap();

            // store the new booster instance
            booster_vec.push(bst_updated);
        }
    }

    async fn make_xg_data_no_labels(zipped_data: Vec<Vec<Vec<f64>>>) -> DMatrix {
        let mut tabular_like_data_vec = Vec::new();

        for tile in zipped_data.iter() {
            let b1 = tile.get(0).unwrap();
            let b2 = tile.get(1).unwrap();
            let b3 = tile.get(2).unwrap();
            let b4 = tile.get(3).unwrap();

            for i in 0..b1.len() {
                let e1 = b1.get(i).unwrap();
                let e2 = b2.get(i).unwrap();
                let e3 = b3.get(i).unwrap();
                let e4 = b4.get(i).unwrap();

                let row = vec![e1.to_owned(), e2.to_owned(), e3.to_owned(), e4.to_owned()];
                tabular_like_data_vec.extend_from_slice(&row);
            }
        }

        let n_rows = tabular_like_data_vec.len() / 4;

        let data_arr_2d =
            Array2::from_shape_vec((n_rows as usize, 4), tabular_like_data_vec).unwrap();

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let xg_matrix = DMatrix::from_col_major_f64(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            512 * 512 * 77,
            4 as usize,
        )
        .unwrap();

        // set labels
        // TODO: make more generic

        xg_matrix
    }

    /// NOTE: This function assumes, that the last column is the target.
    // TODO: change to index argument?
    async fn make_xg_data(
        reservoirs: Vec<Vec<f64>>,
        capacity: usize,
        forward_map: &mut HashMap<String, i32>,
        true_distribution_map: &mut HashMap<String, i32>,
    ) -> DMatrix {
        let mut tabular_like_data_vec = Vec::new();

        let streamed_reservoirs: Vec<_> = reservoirs
            .clone()
            .into_iter()
            .map(|band_reservoir| futures::stream::iter(band_reservoir))
            .collect();

        let streamed_reservoirs_vec = StreamVectorZip::new(streamed_reservoirs);

        let mut rows: Vec<_> = streamed_reservoirs_vec.collect().await;

        let n_cols = rows.get(0).unwrap().len();

        let mut class_counter = 0;
        // let mut target_hashmap = HashMap::new();

        let mut target_vec = Vec::new();
        for row in rows.iter_mut() {
            let target_value = row.pop().unwrap();

            if forward_map.contains_key(format!("{target_value}").as_str()) == false {
                forward_map.insert(format!("{target_value}"), class_counter);
                class_counter += 1;
            }

            *true_distribution_map
                .entry(format!("{target_value}"))
                .or_insert(0) += 1;

            target_vec.push(target_value);
            tabular_like_data_vec.extend_from_slice(&row);
        }

        // we need to remap the target values to [0, num_classes) for xgboost.
        // otherwise it cant perform multi-class classification.

        // println!("target vec: {:?}", target_hashmap);

        let mut target_vec_remapped = Vec::new();
        for target in target_vec.iter() {
            let target_value = forward_map.get(format!("{target}").as_str()).unwrap();
            target_vec_remapped.push(*target_value);
            // *true_counts.entry(format!("{target}")).or_insert(0) += 1;
        }

        assert_eq!(target_vec_remapped.len(), target_vec.len());

        let data_arr_2d =
            Array2::from_shape_vec((capacity, n_cols - 1), tabular_like_data_vec).unwrap();

        // define information needed for xgboost
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
            n_cols - 1 as usize,
        )
        .unwrap();

        // set labels
        // TODO: make more generic

        let lbls: Vec<f32> = target_vec_remapped
            .iter()
            .map(|elem| *elem as f32)
            .collect();
        xg_matrix.set_labels(lbls.as_slice()).unwrap(); // <- here we need the remapped target values
        xg_matrix
    }

    async fn generate_reservoir<'a>(
        zipped_data: &Vec<Vec<Vec<f64>>>,
        tile_size: usize,
        capacity: usize,
    ) -> Vec<Vec<f64>> {
        let mut i = 0;

        let num_of_bands = zipped_data.get(0).unwrap().len();

        // we need a store for each band's reservoir
        let mut vec_of_reservoirs = Vec::new();
        for _ in 0..num_of_bands {
            let band_reservoir: Vec<f64> = Vec::new();
            vec_of_reservoirs.push(band_reservoir);
        }

        let uniform_rand = generate_uniform_rng(0.0, 1.0);

        let mut w = (uniform_rand.ln() / capacity as f64).exp();

        for (tile_counter, tile) in zipped_data.iter().enumerate() {
            // check if next element is in current tile or we can skip this tile
            if elem_in_this_tile(i, tile_counter, tile_size) == false {
                continue;
            }

            // initial fill of the reservoir
            if i < capacity {
                while i < capacity {
                    // we need to "leave" this tile and go to the next. the index exceeds this tile's bounds.
                    if i >= (tile_counter + 1) * (tile_size * tile_size) {
                        break;
                    }

                    // define the index within the tile
                    let idx = i - tile_counter * (tile_size * tile_size);

                    // fill the reservoirs with data
                    for (j, reservoir) in vec_of_reservoirs.iter_mut().enumerate() {
                        let band = tile.get(j).unwrap();
                        let elem = band.get(idx).unwrap().to_owned();
                        reservoir.push(elem);
                    }

                    i = i + 1;
                }
            }
            // consecutive fill of the reservoir with random elements
            while i < (tile_counter + 1) * (tile_size * tile_size) {
                let step = Uniform::new(0, capacity);
                let mut rng = rand::thread_rng();
                let idx_swap_elem = step.sample(&mut rng);

                let idx_this_tile = i - (tile_counter * (tile_size * tile_size));

                // change elements in the reservoir
                for (j, reservoir) in vec_of_reservoirs.iter_mut().enumerate() {
                    let next_band_element =
                        tile.get(j).unwrap().get(idx_this_tile).unwrap().to_owned();
                    reservoir.push(next_band_element);
                    reservoir.swap_remove(idx_swap_elem);
                }

                let uniform_rand = generate_uniform_rng(0.0, 1.0);

                let rand_step = (uniform_rand.ln() / (1.0 - w).ln()).floor();

                let uniform_rand = generate_uniform_rng(0.0, 1.0);

                w = w * (uniform_rand.ln() / capacity as f64).exp();

                i = i + 1 + rand_step as usize;
            }
        }

        vec_of_reservoirs
    }

    /// Is the i-th element in the j-th tile?
    /// True if i is less than i+1 * tile_size.
    fn elem_in_this_tile(i: usize, tile_counter: usize, tile_size: usize) -> bool {
        let result = i < (tile_counter + 1) * (tile_size * tile_size);
        result
    }

    fn generate_uniform_rng(from: f64, to: f64) -> f64 {
        let step = Uniform::new(from, to);
        let mut rng = rand::thread_rng();
        let choice: f64 = step.sample(&mut rng);
        choice
    }

    async fn predict(booster_model: Booster) -> Result<Vec<f32>, xgboost_bindings::XGBError> {
        let paths = [
            "s2_10m_de_marburg/b02.tiff",
            "s2_10m_de_marburg/b03.tiff",
            "s2_10m_de_marburg/b04.tiff",
            "s2_10m_de_marburg/b08.tiff",
        ];

        // define reservoir size

        // how many rounds should be trained?
        // needs to be a power of 2
        // TODO: remove this parameter?
        let tile_size = 512;

        // setup data/model cache

        // do geoengine magic
        let zipped_data: Vec<Vec<Vec<f64>>> = zip_bands_to_tiles(&paths, tile_size).await;

        // make xg compatible, trainable datastructure
        let xg_matrix = make_xg_data_no_labels(zipped_data).await;

        println!("xg_matrix: {:?}", &xg_matrix);

        let mut o: u64 = 0;
        let shp = &[512 as u64 * 512 as u64, 4];
        let result = booster_model.predict_from_dmat(&xg_matrix, shp, &mut o);
        result
    }

    #[test]
    fn xg_test() {
        let data_arr_2d = arr2(&[
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
        ]);

        let target_vec = arr2(&[
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
            [180.0],
        ]);

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let mut xg_matrix = DMatrix::from_col_major_f64(
            data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            20,
            4,
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
            [1.0, 6.0, 10.0, 16.0],
            [1.0, 4.0, 10.0, 16.0],
            [1.0, 7.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 4.0, 10.0, 16.0],
            [1.0, 6.0, 10.0, 16.0],
            [1.0, 3.0, 10.0, 16.0],
            [1.0, 4.0, 10.0, 16.0],
            [1.0, 4.0, 10.0, 16.0],
            [1.0, 8.0, 10.0, 16.0],
            [1.0, 4.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 2.0, 10.0, 16.0],
            [1.0, 5.0, 10.0, 16.0],
            [1.0, 6.0, 10.0, 16.0],
            [1.0, 4.0, 10.0, 16.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]);

        let strides_ax_0 = test_data_arr_2d.strides()[0] as usize;
        let strides_ax_1 = test_data_arr_2d.strides()[1] as usize;
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;

        // get xgboost style matrices
        let mut test_data = DMatrix::from_col_major_f64(
            test_data_arr_2d.as_slice_memory_order().unwrap(),
            byte_size_ax_0,
            byte_size_ax_1,
            20,
            4,
        )
        .unwrap();

        let result = bst.predict(&test_data).unwrap();
        println!("result: {:?}", result);
    }
}
