use futures::{StreamExt};
use geoengine_datatypes::{primitives::{SpatialPartition2D}, raster::{GridOrEmpty}};
use crate::engine::{QueryContext, QueryRectangle, RasterQueryProcessor};
use pyo3::{types::{PyModule}};
use ndarray::{Array2, Axis, stack, ArrayBase, OwnedRepr, Dim};
use numpy::{PyArray};
use crate::util::Result;

#[allow(clippy::too_many_arguments)]
pub async fn imseg_fit<C: QueryContext>(
    processor_ir_016: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_ir_039: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_ir_087: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_ir_097: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_ir_108: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_ir_120: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_ir_134: Box<dyn RasterQueryProcessor<RasterType = f32>>,
    processor_truth: Box<dyn RasterQueryProcessor<RasterType = u8>>,
    query_rect: QueryRectangle<SpatialPartition2D>,
    query_ctx: C,
) -> Result<()>
{

    //For some reason we need that now...
    pyo3::prepare_freethreaded_python();
    let gil = pyo3::Python::acquire_gil();
    let py = gil.python();


    let py_mod = PyModule::from_code(py, include_str!("tf_v2.py"),"filename.py", "modulename").unwrap();

        
    let tile_stream_ir_016 = processor_ir_016.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_ir_039 = processor_ir_039.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_ir_087 = processor_ir_087.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_ir_097 = processor_ir_097.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_ir_108 = processor_ir_108.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_ir_120 = processor_ir_120.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_ir_134 = processor_ir_134.raster_query(query_rect, &query_ctx).await?;
    let tile_stream_truth = processor_truth.raster_query(query_rect, &query_ctx).await?;

    let mut final_stream = tile_stream_ir_016.zip(tile_stream_ir_039.zip(tile_stream_ir_087.zip(tile_stream_ir_097.zip(tile_stream_ir_108.zip(tile_stream_ir_120.zip(tile_stream_ir_134.zip(tile_stream_truth)))))));
    let mut i = 0;
    while let Some((ir_016, (ir_039, (ir_087, (ir_097, (ir_108, (ir_120, (ir_134, truth)))))))) = final_stream.next().await {
        match (ir_016, ir_039, ir_087, ir_097, ir_108, ir_120, ir_134, truth) {
            (Ok(ir_016), Ok(ir_039), Ok(ir_087), Ok(ir_097), Ok(ir_108), Ok(ir_120), Ok(ir_134), Ok(truth)) => {
                dbg!(&ir_016.time);
                dbg!(&truth.time);
                dbg!(&ir_016.global_geo_transform);
                dbg!(&truth.global_geo_transform);
                dbg!(&ir_016.tile_position);
                dbg!(&truth.tile_position);
                match (ir_016.grid_array, ir_039.grid_array, ir_087.grid_array, ir_097.grid_array, ir_108.grid_array, ir_120.grid_array, ir_134.grid_array, truth.grid_array) {
                    (GridOrEmpty::Grid(grid_016), GridOrEmpty::Grid(grid_039),  GridOrEmpty::Grid(grid_087),  GridOrEmpty::Grid(grid_097), GridOrEmpty::Grid(grid_108), GridOrEmpty::Grid(grid_120), GridOrEmpty::Grid(grid_134), GridOrEmpty::Grid(grid_truth)) => {

                        let tile_size = grid_016.shape.shape_array;

                        let (arr_img, arr_truth) = create_arrays_from_data(grid_016.data, grid_039.data, grid_087.data, grid_097.data, grid_108.data, grid_120.data, grid_134.data, grid_truth.data, tile_size);
                        
                        

                        let py_img = PyArray::from_owned_array(py, arr_img);
                        let py_truth = PyArray::from_owned_array(py, arr_truth);
                        let _result = py_mod.call("test", (py_img, py_truth, i), None).unwrap();
                        i = i + 1;
                        
                        
                    },
                    _ => {
                        println!("Some were empty");
                        
                    }
                }
            },
            _ => {
                println!("An error occured!");
            }
        }

    }

    
    
    Ok(())
    
}

/// Creates batches used for training the model from the vectors of the rasterbands
fn create_arrays_from_data<T, U>(data_1: Vec<T>, data_2: Vec<T>, data_3: Vec<T>, data_4: Vec<T>, data_5: Vec<T>, data_6: Vec<T>, data_7: Vec<T>, data_8: Vec<U>, tile_size: [usize;2]) -> (ArrayBase<OwnedRepr<T>, Dim<[usize; 4]>>, ArrayBase<OwnedRepr<U>, Dim<[usize; 4]>>) 
where 
T: Clone + std::marker::Copy, 
U: Clone {

    let arr_1: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_1)
            .unwrap();
            let arr_2: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_2)
            .unwrap();
            let arr_3: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_3)
            .unwrap();
            let arr_4: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_4)
            .unwrap()
            .to_owned();
            let arr_5: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_5)
            .unwrap()
            .to_owned();
            let arr_6: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_6)
            .unwrap()
            .to_owned();
            let arr_7: ndarray::Array2<T> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_7)
            .unwrap()
            .to_owned();
            let arr_8: ndarray::Array4<U> = 
            Array2::from_shape_vec((tile_size[0], tile_size[1]), data_8)
            .unwrap()
            .to_owned().insert_axis(Axis(0)).insert_axis(Axis(3));
            let arr_res: ndarray::Array<T, _> = stack(Axis(2), &[arr_1.view(),arr_2.view(),arr_3.view(), arr_4.view(), arr_5.view(), arr_6.view(), arr_7.view()]).unwrap().insert_axis(Axis(0));

            (arr_res, arr_8)
}



#[cfg(test)]
mod tests {
    use geoengine_datatypes::{dataset::{DatasetId, InternalDatasetId}, hashmap, primitives::{Coordinate2D, TimeInterval, SpatialResolution,  Measurement, TimeGranularity, TimeInstance, TimeStep}, raster::{GeoTransform, RasterDataType}, raster::{TilingSpecification}, spatial_reference::{SpatialReference, SpatialReferenceAuthority, SpatialReferenceOption}};
 

    use geoengine_datatypes::{util::Identifier, raster::{RasterPropertiesEntryType}};
    use std::{path::PathBuf};
    use crate::{engine::{MockExecutionContext,MockQueryContext, RasterOperator, RasterResultDescriptor}, source::{FileNotFoundHandling, GdalDatasetParameters, GdalMetaDataRegular, GdalSource, GdalSourceParameters,GdalMetadataMapping, GdalSourceTimePlaceholder, TimeReference, GdalDatasetGeoTransform}};
    use crate::processing::{expression::{Expression, ExpressionParams, ExpressionSources}, meteosat::{offset_key, slope_key, satellite_key, channel_key, radiance::{Radiance, RadianceParams}, temperature::{Temperature, TemperatureParams}, reflectance::{Reflectance, ReflectanceParams}}};
    use crate::engine::SingleRasterSource;

    use super::*;

    #[tokio::test]
    async fn imseg_meaningless() {
        let ctx = MockQueryContext::default();
        //tile size has to be a power of 2
        let tiling_specification =
            TilingSpecification::new(Coordinate2D::default(), [512, 512].into());

        let query_spatial_resolution = SpatialResolution::new(3000.4, 3000.4).unwrap();

        let query_bbox = SpatialPartition2D::new((-802607.8468561172485352, 5108186.3898038864135742).into(), (1498701.3813257217407227, 3577980.7752370834350586).into()).unwrap();
        let no_data_value = Some(0.);
        
        let ir_016 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_016___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: (-5570248.477339744567871,5570248.477339744567871).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3712,
                height: 3712,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_016___-000001___-%Y%m%d%H%M-C_".to_string(),
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }
        };
        let ir_039 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_039___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
             result_descriptor: RasterResultDescriptor{
                 data_type: RasterDataType::I16,
                 spatial_reference: SpatialReferenceOption::
                     SpatialReference(SpatialReference{
                         authority: SpatialReferenceAuthority::SrOrg,
                         code: 81,
                     }),
                 measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                 no_data_value,
             },
             params: GdalDatasetParameters{
                 file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                 rasterband_channel: 1,
                 geo_transform: GdalDatasetGeoTransform{
                     origin_coordinate: (-5570248.477339744567871,5570248.477339744567871).into(),
                     x_pixel_size: 3000.403165817260742,
                     y_pixel_size: -3000.403165817260742, 
                 },
                 width: 3712,
                 height: 3712,
                 file_not_found_handling: FileNotFoundHandling::NoData,
                 no_data_value,
                 properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                 gdal_open_options: None,
                 gdal_config_options: None,
             },
             //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
             //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_039___-000001___-%Y%m%d%H%M-C_".to_string(),
             start: TimeInstance::from_millis(1072917000000).unwrap(),
             step: TimeStep{
                 granularity: TimeGranularity::Minutes,
                step: 15,
            },

        };

        let ir_087 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_087___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: (-5570248.477, 5570248.477).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3712,
                height: 3712,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_087___-000001___-%Y%m%d%H%M-C_".to_string(),
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }

        };

        let ir_097 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_097___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: (-5570248.477, 5570248.477).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3712,
                height: 3712,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_097___-000001___-%Y%m%d%H%M-C_".to_string(),
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }

        };


        let ir_108 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_108___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: (-5570248.477, 5570248.477).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3712,
                height: 3712,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_108___-000001___-%Y%m%d%H%M-C_".to_string(),
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }

        };

        let ir_120 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_120___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: (-5570248.477, 5570248.477).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3712,
                height: 3712,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_120___-000001___-%Y%m%d%H%M-C_".to_string(),
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }

        };

        let ir_134 = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_134___-000001___-%Y%m%d%H%M-C_".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::I16,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Continuous{
                    measurement: "raw".to_string(),
                    unit: None
                },
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("/mnt/panq/dbs_geo_data/satellite_data/msg_seviri/%%%_TIME_FORMATED_%%%"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: (-5570248.477, 5570248.477).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3712,
                height: 3712,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: Some(vec![GdalMetadataMapping::identity(offset_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(slope_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(channel_key(), RasterPropertiesEntryType::Number), GdalMetadataMapping::identity(satellite_key(), RasterPropertiesEntryType::Number)]),
                gdal_open_options: None,
                gdal_config_options: None,
            },
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/%Y%m%d_%H%M/H-000-MSG3__-MSG3________-IR_134___-000001___-%Y%m%d%H%M-C_".to_string(),
            start: TimeInstance::from_millis(1072917000000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }

        };

        let claas = GdalMetaDataRegular{
            time_placeholders: hashmap! {
                "%%%_TIME_FORMATED_%%%".to_string() => GdalSourceTimePlaceholder {
                    format: "%Y/%m/%d/CMAin%Y%m%d%H%M00305SVMSG01MD".to_string(),
                    reference: TimeReference::Start,

                }
            },
            result_descriptor: RasterResultDescriptor{
                data_type: RasterDataType::U8,
                spatial_reference: SpatialReferenceOption::
                    SpatialReference(SpatialReference{
                        authority: SpatialReferenceAuthority::SrOrg,
                        code: 81,
                    }),
                measurement: Measurement::Unitless,
                no_data_value,
            },
            params: GdalDatasetParameters{
                file_path: PathBuf::from("NETCDF:\"/mnt/panq/dbs_geo_data/satellite_data/CLAAS-2/level2/%%%_TIME_FORMATED_%%%.nc\":cma"),
                rasterband_channel: 1,
                geo_transform: GdalDatasetGeoTransform{
                    origin_coordinate: ( -5456233.41938636, 5456233.41938636).into(),
                    x_pixel_size: 3000.403165817260742,
                    y_pixel_size: -3000.403165817260742, 
                },
                width: 3636,
                height: 3636,
                file_not_found_handling: FileNotFoundHandling::NoData,
                no_data_value,
                properties_mapping: None,
                gdal_open_options: None,
                gdal_config_options: None,
            },
            //placeholder: "%%%_TIME_FORMATED_%%%".to_string(),
            //time_format: "%Y/%m/%d/CMAin%Y%m%d%H%M00305SVMSG01MD".to_string(),
            start: TimeInstance::from_millis(1356994800000).unwrap(),
            step: TimeStep{
                granularity: TimeGranularity::Minutes,
                step: 15,
            }

        };

        let mut mc = MockExecutionContext::default();
        mc.tiling_specification = tiling_specification;

        let id_ir_016 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_016 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_016.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_016, Box::new(ir_016.clone()));

        let rad_ir_016 = RasterOperator::boxed(Radiance{
            params: RadianceParams{
                
            },
            sources: SingleRasterSource{
                raster: op_ir_016
            }
        });
        let ref_ir_016 = RasterOperator::boxed(Reflectance {
            params: ReflectanceParams::default(),
            sources: SingleRasterSource{
                raster: rad_ir_016,
            }
        });

        let exp_ir_016 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-0.15282721917305925)/0.20047640300325603".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: ref_ir_016.clone(),
            b: Some(ref_ir_016),
            c: None,
        }
        });
        
        let proc_ir_016 = exp_ir_016
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();

        let id_ir_039 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_039 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_039.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_039, Box::new(ir_039.clone()));

        let temp_ir_039 = RasterOperator::boxed(Temperature{
            params: TemperatureParams::default(),
            sources: SingleRasterSource{
                raster: op_ir_039
            }
        });

        let exp_ir_039 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-276.72667474831303)/15.982918482298778".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: temp_ir_039.clone(),
            b: Some(temp_ir_039),
            c: None,
        }
        });
        let proc_ir_039 = exp_ir_039
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();

        let id_ir_087 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_087 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_087.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_087, Box::new(ir_087.clone()));

        let temp_ir_087 = RasterOperator::boxed(Temperature{
            params: TemperatureParams::default(),
            sources: SingleRasterSource{
                raster: op_ir_087
            }
        });

        let exp_ir_087 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-267.92274094012157)/15.763409602725156".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: temp_ir_087.clone(),
            b: Some(temp_ir_087),
            c: None,
        }
        });
        let proc_ir_087 = exp_ir_087
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();

        let id_ir_097 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_097 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_097.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_097, Box::new(ir_097.clone()));

        let temp_ir_097 = RasterOperator::boxed(Temperature{
            params: TemperatureParams::default(),
            sources: SingleRasterSource{
                raster: op_ir_097
            }
        });

        let exp_ir_097 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-245.4006454137375)/9.644958714922186".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: temp_ir_097.clone(),
            b: Some(temp_ir_097),
            c: None,
        }
        });
        let proc_ir_097 = exp_ir_097
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();
        
        let id_ir_108 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_108 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_108.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_108, Box::new(ir_108.clone()));

        let temp_ir_108 = RasterOperator::boxed(Temperature{
            params: TemperatureParams::default(),
            sources: SingleRasterSource{
                raster: op_ir_108
            }
        });

        let exp_ir_108 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-269.95727803541155)/16.92469947004928".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: temp_ir_108.clone(),
            b: Some(temp_ir_108),
            c: None,
        }
        });
        let proc_ir_108 = exp_ir_108
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();


        let id_ir_120 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_120 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_120.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_120, Box::new(ir_120.clone()));

        let temp_ir_120 = RasterOperator::boxed(Temperature{
            params: TemperatureParams::default(),
            sources: SingleRasterSource{
                raster: op_ir_120
            }
        });
        let exp_ir_120 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-268.69063154766155)/16.963088951563815".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: temp_ir_120.clone(),
            b: Some(temp_ir_120),
            c: None,
        }
        });
        let proc_ir_120 = exp_ir_120
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();

        let id_ir_134 = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_ir_134 = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_ir_134.clone(),
            }
        });

        
        mc.add_meta_data(id_ir_134, Box::new(ir_134.clone()));

        let temp_ir_134 = RasterOperator::boxed(Temperature{
            params: TemperatureParams::default(),
            sources: SingleRasterSource{
                raster: op_ir_134
            }
        });
        let exp_ir_134 = RasterOperator::boxed(Expression{
            params: ExpressionParams { expression: "(A-252.1465931705522)/11.171453493090551".to_string(), output_type: RasterDataType::F32, output_no_data_value: no_data_value.unwrap(), output_measurement: Some(Measurement::Continuous{
                measurement: "raw".to_string(),
                unit: None,
            })
        },
        sources: ExpressionSources{
            a: temp_ir_134.clone(),
            b: Some(temp_ir_134),
            c: None,
        }
        });
        let proc_ir_134 = exp_ir_134
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_f32().unwrap();

        let id_claas = DatasetId::Internal{
            dataset_id: InternalDatasetId::new(),
        };

        let op_claas = Box::new(GdalSource{
            params: GdalSourceParameters{
                dataset: id_claas.clone(),
            }
        });

        mc.add_meta_data(id_claas, Box::new(claas.clone()));

        
        let proc_claas = op_claas
        .initialize(&mc).await.unwrap().query_processor().unwrap().get_u8().unwrap();


        let _x = imseg_fit(proc_ir_016, proc_ir_039,proc_ir_087, proc_ir_097, proc_ir_108, proc_ir_120, proc_ir_134, proc_claas, QueryRectangle {
            spatial_bounds: query_bbox,
            time_interval: TimeInterval::new(1357048800000, 1357048800000 + 1000)
                .unwrap(),
            spatial_resolution: query_spatial_resolution,
        }, ctx,
    ).await.unwrap();
    }
}
