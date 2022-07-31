#[cfg(test)]
mod tests {
    use rand::distributions::{Distribution, Uniform};

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
    use ndarray::Array2;
    use xgboost_bindings::{Booster, DMatrix};

    use std::collections::{btree_map::Entry, BTreeMap, HashMap};
    use std::f64;
    use std::mem::{self};
    use std::path::PathBuf;

    fn get_gdal_config_metadata(path: &str) -> GdalMetaDataRegular {
        let no_data_value = Some(-1000.0);

        GdalMetaDataRegular {
            result_descriptor: RasterResultDescriptor {
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::SpatialReference(SpatialReference::new(
                    SpatialReferenceAuthority::Epsg,
                    32632,
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
                    origin_coordinate: (474_112.0, 5_646_336.0).into(),
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
            start: TimeInstance::from_millis(1_072_917_000_000).unwrap(),
            step: TimeStep {
                granularity: TimeGranularity::Minutes,
                step: 15,
            },
        }
    }

    async fn initialize_operator(
        gcm: GdalMetaDataRegular,
        tile_size: usize,
    ) -> Box<dyn RasterQueryProcessor<RasterType = i16>> {
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
            .get_i16()
            .unwrap();

        rqp_gt
    }

    async fn get_band_data(
        rqp_gt: Box<dyn RasterQueryProcessor<RasterType = i16>>,
    ) -> Vec<Vec<f64>> {
        let ctx = MockQueryContext::test_default();

        let query_bbox = SpatialPartition2D::new(
            (474_112.000, 5_646_336.000).into(),
            (522_752.000, 5_612_026.000).into(),
        )
        .unwrap();
        let query_spatial_resolution = SpatialResolution::new(10.0, 10.0).unwrap();

        let qry_rectangle = QueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1_590_969_600_000, 1_590_969_600_000).unwrap(),
            spatial_resolution: query_spatial_resolution,
        };

        let mut stream = rqp_gt.raster_query(qry_rectangle, &ctx).await.unwrap();

        let mut buffer_proc: Vec<Vec<f64>> = Vec::new();

        while let Some(processor) = stream.next().await {
            match processor.unwrap().grid_array {
                GridOrEmpty::Grid(processor) => {
                    let data = &processor.data;
                    // TODO: make more generic
                    let data_mapped: Vec<f64> = data.iter().map(|elem| f64::from(*elem)).collect();
                    buffer_proc.push(data_mapped);
                }
                GridOrEmpty::Empty(_) => {
                    buffer_proc.push(vec![]);
                }
            }
        }

        buffer_proc
    }

    /// This function calculates the maximum possible reservoir size for the given parameters.
    /// # Arguments
    /// `max_mem_cap`: How big the maximum memory capacity can be (in given unit).
    /// unit: What the unit of the mem cap is. -> Megabytes etc.
    /// `n_bands`: How many bands are read.
    /// `type_size`: The size of the elements in the bands.
    /// types: A vector of type sizes. Contains information if bands are of different types. TODO
    //TODO: what if reservoir is bigger than dataset
    fn calculate_reservoir_size(
        max_mem_cap: usize,
        unit: &str,
        n_bands: usize,
        type_size: usize,
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

    #[tokio::test]
    async fn training_workflow() {
        // define data to be used
        // the target should be last
        let paths = [
            "s2_10m_de_marburg/b02.tiff",
            "s2_10m_de_marburg/b03.tiff",
            "s2_10m_de_marburg/b04.tiff",
            "s2_10m_de_marburg/b08.tiff",
            // put target band at the last slot
            "s2_10m_de_marburg/target.tiff",
        ];

        // define reservoir size
        // TODO: bigger reservoir size than dataset
        let capacity = calculate_reservoir_size(1, "gb", paths.len(), mem::size_of::<f64>());

        // how many rounds should be trained?
        let training_rounds = 2;

        // needs to be a power of 2
        // TODO: remove this parameter?
        let tile_size = 512;

        // setup data/model cache
        let mut booster_vec: Vec<Booster> = Vec::new();
        let mut matrix_vec: Vec<DMatrix> = Vec::new();

        // do geoengine magic
        println!("starting geoengine magic");

        // this contains Tiles[Bands[Pixelvalues]]
        let tiles_bands_pixel_vec: Vec<Vec<Vec<f64>>> =
            get_tiles_bands_pixels_vec(&paths, tile_size).await;

        // extract the target band to analyze the unique classes.
        let target_vec: Vec<f64> = tiles_bands_pixel_vec
            .iter()
            .flat_map(|tile| tile[paths.len() - 1].clone())
            .collect();

        // we need a forward map to help xg boost train the classification. The classes need to be in [0, n_classes).
        // we need a backward map to give the predictions back in the original coding.
        // true distribution is only to verify data after training.
        let forward_map = get_hashmaps(&target_vec);

        println!("training with a reservoir size of {:?}", capacity);
        for _ in 0..training_rounds {
            // generate and fill a reservoir
            println!("generating reservoir");

            let mut reservoirs = generate_reservoir(&tiles_bands_pixel_vec, tile_size, capacity);

            // make xg compatible, trainable datastructure
            println!("generating xg matrix");
            let xg_matrix = make_xg_data(&mut reservoirs, &forward_map);

            // start the training process
            // TODO: num_rounds implementieren
            train_model(&mut booster_vec, &mut matrix_vec, xg_matrix);
        }

        println!("done");
    }

    /// This function takes a slice of paths to `band_i.tif` files and turns them into a vector of zipped, tiled data.
    /// Each band is tiled in the beginning. The elements per tile are then zipped together, such that a tabular style
    /// result is returned.
    /// The result contains all tiles of all bands of the provided data.
    /// The structure looks like this:
    /// `Zipped_data`[
    ///     `Tile_1`[
    ///        `Band_1`[`elem1,...,elem_tilesize^2`],
    ///        ...,
    ///        `Band_n`[`elem1,...,elem_tilesize^2`]
    ///           ],
    ///     ...,
    ///    `Tile_n`[
    ///        `Band_1`[`elem1,...,elem_tilesize^2`],
    ///        ...,
    ///        `Band_n`[`elem1,...,elem_tilesize^2`]
    ///           ]
    ///   ]

    async fn get_tiles_bands_pixels_vec(paths: &[&str], tile_size: usize) -> Vec<Vec<Vec<f64>>> {
        let mut bands: Vec<Vec<Vec<f64>>> = vec![];

        // load each band given by distinct .tif files
        for path in paths.iter() {
            let gcm = get_gdal_config_metadata(path);
            let init_op = initialize_operator(gcm, tile_size).await;
            let buffer_proc = get_band_data(init_op).await;
            bands.push(buffer_proc);
        }

        let streamed_bands: Vec<_> = bands.into_iter().map(futures::stream::iter).collect();

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
        if booster_vec.is_empty() {
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

    /// NOTE: This function assumes, that the last column is the target.
    // TODO: change to index argument?
    fn make_xg_data(
        reservoirs: &mut Vec<Vec<f64>>,
        forward_map: &BTreeMap<String, i32>,
    ) -> DMatrix {
        // assuming the last column is the target
        let target_vec = reservoirs.remove(reservoirs.len() - 1);
        let n_cols = reservoirs.len();
        let n_rows = reservoirs[0].len();

        // build a sequential vector of the data by iterating along the columns for every row
        // so the data layout should be like: [[r1,c1], [r1,c2], [r1,...], [r2,c1], [r2,c2], ...]
        let mut sequential_data = Vec::new();
        (0..n_rows).for_each(|row| {
            (0..n_cols).for_each(|col| {
                sequential_data.push(reservoirs[col][row]);
            });
        });

        // we need to remap the target values to [0, num_classes) for xgboost.
        // otherwise it cant perform multi-class classification.
        let mut labels = Vec::new();
        for target_val in &target_vec {
            let target_value = forward_map.get(format!("{target_val}").as_str()).unwrap();
            labels.push(*target_value as f32);
        }

        assert_eq!(labels.len(), target_vec.len());

        let data_arr_2d = Array2::from_shape_vec((n_rows, n_cols), sequential_data).unwrap();

        // define information needed for xgboost
        let strides_ax_0 = data_arr_2d.strides()[0] as usize;
        println!("strides_ax_0: {}", strides_ax_0);
        let strides_ax_1 = data_arr_2d.strides()[1] as usize;
        println!("strides_ax_1: {}", strides_ax_1);
        let byte_size_ax_0 = mem::size_of::<f64>() * strides_ax_0;
        println!("byte_size_ax_0: {}", byte_size_ax_0);
        let byte_size_ax_1 = mem::size_of::<f64>() * strides_ax_1;
        println!("byte_size_ax_1: {}", byte_size_ax_1);

        let data_slice_mem_order = data_arr_2d.as_slice_memory_order().unwrap();

        // get xgboost style matrices
        let mut xg_matrix = DMatrix::from_col_major_f64(
            data_slice_mem_order,
            byte_size_ax_0,
            byte_size_ax_1,
            n_rows,
            n_cols,
        )
        .unwrap();

        // set labels
        xg_matrix.set_labels(labels.as_slice()).unwrap(); // <- here we need the remapped target values
        xg_matrix
    }

    /// Is the i-th element in the j-th tile?
    /// True if i is less than (i+1) * `tile_size`.
    /// For example: The first 256 elements belong to the 0-th tile, if i < 256 == 1 * 16 * 16.
    fn is_elem_in_this_tile(i: usize, tile_counter: usize, tile_size: usize) -> bool {
        i < (tile_counter + 1) * usize::pow(tile_size, 2)
    }

    fn get_hashmaps(zipped_data_target: &Vec<f64>) -> BTreeMap<String, i32> {
        let mut forward_map = BTreeMap::new();
        let mut num_classes = 0;
        for elem in zipped_data_target {
            if let Entry::Vacant(e) = forward_map.entry(format!("{elem}")) {
                e.insert(num_classes);
                num_classes += 1;
            }
        }

        forward_map
    }

    fn generate_reservoir(
        zipped_data: &[Vec<Vec<f64>>],
        tile_size: usize,
        capacity: usize,
    ) -> Vec<Vec<f64>> {
        // reservoir sampling algorithm l
        let mut i = 0;

        let num_of_bands = zipped_data.get(0).unwrap().len();

        // we need a store for each band's reservoir
        let mut vec_of_reservoirs = Vec::new();
        for _ in 0..num_of_bands {
            let band_reservoir: Vec<f64> = Vec::new();
            vec_of_reservoirs.push(band_reservoir);
        }

        let mut tile_counter = 0;

        // filling phase
        'outer: while let Some(tile) = zipped_data.get(tile_counter) {
            for pxl in 0..usize::pow(tile_size, 2) {
                // fill the reservoirs with data
                for (j, reservoir) in vec_of_reservoirs.iter_mut().enumerate() {
                    let band = tile.get(j).unwrap();
                    let elem = band.get(pxl).unwrap().to_owned();
                    reservoir.push(elem);
                }

                if i == capacity - 1 {
                    break 'outer;
                }
                i += 1;
            }

            tile_counter += 1;
        }

        // random phase
        let mut rng = rand::thread_rng();
        let unit_interval = Uniform::from(0.0f64..1.0f64);
        let capacity_interval = Uniform::from(0..capacity);

        let mut w = (unit_interval.sample(&mut rng).ln() / capacity as f64).exp();
        while let Some(tile) = zipped_data.get(tile_counter) {
            loop {
                i = i
                    + 1
                    + ((unit_interval.sample(&mut rng).ln() / (1.0 - w).ln()).floor() as usize);
                if !is_elem_in_this_tile(i, tile_counter, tile_size) {
                    tile_counter += 1;
                    break;
                }
                // select index to swap on
                let rnd_idx = capacity_interval.sample(&mut rng);
                let elem_idx_in_this_tile = i - (tile_counter * usize::pow(tile_size, 2));
                assert!(elem_idx_in_this_tile < usize::pow(tile_size, 2));

                // swap element
                for (j, reservoir) in vec_of_reservoirs.iter_mut().enumerate() {
                    let next_band_element = tile
                        .get(j)
                        .unwrap()
                        .get(elem_idx_in_this_tile)
                        .unwrap()
                        .to_owned();
                    reservoir[rnd_idx] = next_band_element;
                }

                w *= (unit_interval.sample(&mut rng).ln() / capacity as f64).exp();
            }
        }

        vec_of_reservoirs
    }
}
