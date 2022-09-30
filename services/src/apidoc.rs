use crate::api::model::datatypes::{
    BoundingBox2D, Breakpoint, ClassificationMeasurement, Colorizer, ContinuousMeasurement,
    Coordinate2D, DataId, DataProviderId, DatasetId, ExternalDataId, FeatureDataType, LayerId,
    Measurement, Palette, RasterDataType, RasterQueryRectangle, RgbaColor, SpatialPartition2D,
    SpatialReference, SpatialReferenceAuthority, SpatialReferenceOption, SpatialResolution,
    TimeInstance, TimeInterval, VectorDataType,
};
use crate::api::model::operators::{
    PlotResultDescriptor, RasterResultDescriptor, TypedOperator, TypedResultDescriptor,
    VectorColumnInfo, VectorResultDescriptor,
};
use crate::contexts::{SessionId, SimpleSession};
use crate::datasets::listing::{Provenance, ProvenanceOutput};
use crate::datasets::upload::UploadId;
use crate::handlers;
use crate::handlers::tasks::TaskAbortOptions;
use crate::handlers::workflows::{RasterDatasetFromWorkflow, RasterDatasetFromWorkflowResult};
use crate::layers::layer::{
    CollectionItem, Layer, LayerCollection, LayerCollectionListing, LayerListing, Property,
    ProviderLayerCollectionId, ProviderLayerId,
};
use crate::layers::listing::LayerCollectionId;
use crate::projects::{
    ColorParam, DerivedColor, DerivedNumber, LineSymbology, NumberParam, PointSymbology,
    PolygonSymbology, ProjectId, RasterSymbology, STRectangle, StrokeParam, Symbology,
    TextSymbology,
};
use crate::tasks::{TaskFilter, TaskId, TaskListOptions, TaskStatus};
use crate::util::server::VersionInfo;
use crate::util::{apidoc::ServerInfo, IdResponse};
use crate::workflows::workflow::{Workflow, WorkflowId};
use utoipa::openapi::security::{HttpAuthScheme, HttpBuilder, SecurityScheme};
use utoipa::{Modify, OpenApi};

#[derive(OpenApi)]
#[openapi(
    paths(
        crate::util::server::show_version_handler,
        handlers::layers::layer_handler,
        handlers::layers::list_collection_handler,
        handlers::layers::list_root_collections_handler,
        handlers::session::anonymous_handler,
        handlers::session::session_handler,
        handlers::session::session_project_handler,
        handlers::session::session_view_handler,
        handlers::tasks::abort_handler,
        handlers::tasks::list_handler,
        handlers::tasks::status_handler,
        handlers::workflows::dataset_from_workflow_handler,
        handlers::workflows::get_workflow_metadata_handler,
        handlers::workflows::get_workflow_provenance_handler,
        handlers::workflows::load_workflow_handler,
        handlers::workflows::register_workflow_handler,
    ),
    components(
        schemas(
            SimpleSession,

            DataId,
            DataProviderId,
            DatasetId,
            ExternalDataId,
            IdResponse<WorkflowId>,
            LayerId,
            ProjectId,
            SessionId,
            TaskId,
            UploadId,
            WorkflowId,
            ProviderLayerId,
            ProviderLayerCollectionId,
            LayerCollectionId,

            TimeInstance,
            TimeInterval,

            Coordinate2D,
            BoundingBox2D,
            SpatialPartition2D,
            SpatialResolution,
            SpatialReference,
            SpatialReferenceOption,
            SpatialReferenceAuthority,
            Measurement,
            ContinuousMeasurement,
            ClassificationMeasurement,
            STRectangle,

            ProvenanceOutput,
            Provenance,

            VectorDataType,
            FeatureDataType,
            RasterDataType,

            VersionInfo,

            Workflow,
            TypedOperator,
            TypedResultDescriptor,
            PlotResultDescriptor,
            RasterResultDescriptor,
            VectorResultDescriptor,
            VectorColumnInfo,
            RasterDatasetFromWorkflow,
            RasterDatasetFromWorkflowResult,
            RasterQueryRectangle,
            // VectorQueryRectangle,
            // PlotQueryRectangle,

            TaskAbortOptions,
            TaskFilter,
            TaskListOptions,
            TaskStatus,

            Layer,
            LayerListing,
            LayerCollection,
            LayerCollectionListing,
            Property,
            CollectionItem,

            Breakpoint,
            ColorParam,
            Colorizer,
            DerivedColor,
            DerivedNumber,
            LineSymbology,
            NumberParam,
            Palette,
            PointSymbology,
            PolygonSymbology,
            RasterSymbology,
            RgbaColor,
            StrokeParam,
            Symbology,
            TextSymbology,
        ),
    ),
    modifiers(&SecurityAddon, &ApiDocInfo, &ServerInfo),
    external_docs(url = "https://docs.geoengine.io", description = "Geo Engine Docs")
)]
pub struct ApiDoc;

struct SecurityAddon;

impl Modify for SecurityAddon {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        let components = openapi.components.as_mut().unwrap();
        components.add_security_scheme(
            "session_token",
            SecurityScheme::Http(
                HttpBuilder::new()
                    .scheme(HttpAuthScheme::Bearer)
                    .bearer_format("UUID")
                    .description(Some("A valid session token can be obtained via the /anonymous or /login (pro only) endpoints. Alternatively, it can be defined as a fixed value in the Settings.toml file."))
                    .build(),
            ),
        );
    }
}

struct ApiDocInfo;

impl Modify for ApiDocInfo {
    fn modify(&self, openapi: &mut utoipa::openapi::OpenApi) {
        openapi.info.title = "Geo Engine API".to_string();

        openapi.info.contact = Some(
            utoipa::openapi::ContactBuilder::new()
                .name(Some("Geo Engine Developers"))
                .email(Some("dev@geoengine.de"))
                .build(),
        );

        openapi.info.license = Some(
            utoipa::openapi::LicenseBuilder::new()
                .name("Apache 2.0 (pro features excluded)")
                .url(Some(
                    "https://github.com/geo-engine/geoengine/blob/master/LICENSE",
                ))
                .build(),
        );
    }
}
