use std::{
    collections::HashMap,
    ffi::OsStr,
    fs::{self, DirEntry, File},
    io::BufReader,
    path::PathBuf,
};

use crate::error::Result;
use crate::{
    layers::layer::{
        AddLayer, AddLayerCollection, LayerCollectionDefinition, LayerCollectionId, LayerDefinition,
    },
    workflows::registry::WorkflowRegistry,
};
use crate::{layers::storage::LayerDb, util::user_input::UserInput};

use log::{info, warn};

pub async fn add_layers_from_directory<L: LayerDb, W: WorkflowRegistry>(
    layer_db: &mut L,
    workflow_db: &mut W,
    file_path: PathBuf,
) {
    async fn add_layer_from_dir_entry<L: LayerDb, W: WorkflowRegistry>(
        layer_db: &mut L,
        workflow_db: &mut W,
        entry: &DirEntry,
    ) -> Result<()> {
        let def: LayerDefinition =
            serde_json::from_reader(BufReader::new(File::open(entry.path())?))?;

        let workflow_id = workflow_db.register(def.workflow).await?;

        layer_db
            .add_layer_with_id(
                def.id,
                AddLayer {
                    name: def.name,
                    description: def.description,
                    workflow: workflow_id,
                    symbology: def.symbology,
                }
                .validated()?,
            )
            .await?;

        Ok(())
    }

    let dir = fs::read_dir(file_path);
    if dir.is_err() {
        warn!("Skipped adding layers from directory because it can't be read");
        return;
    }
    let dir = dir.expect("checked");

    for entry in dir {
        match entry {
            Ok(entry) if entry.path().extension() == Some(OsStr::new("json")) => {
                match add_layer_from_dir_entry(layer_db, workflow_db, &entry).await {
                    Ok(_) => info!("Added layer from directory entry: {:?}", entry),
                    Err(e) => warn!(
                        "Skipped adding layer from directory entry: {:?} error: {}",
                        entry,
                        e.to_string()
                    ),
                }
            }
            _ => {
                warn!("Skipped adding layer from directory entry: {:?}", entry);
            }
        }
    }
}

pub async fn add_layer_collections_from_directory<L: LayerDb>(db: &mut L, file_path: PathBuf) {
    fn get_layer_collection_from_dir_entry(entry: &DirEntry) -> Result<LayerCollectionDefinition> {
        Ok(serde_json::from_reader(BufReader::new(File::open(
            entry.path(),
        )?))?)
    }

    async fn add_collection_to_db<L: LayerDb>(
        db: &mut L,
        def: &LayerCollectionDefinition,
    ) -> Result<()> {
        let collection = AddLayerCollection {
            name: def.name.clone(),
            description: def.description.clone(),
        }
        .validated()?;

        db.add_collection_with_id(def.id, collection).await?;

        for layer in &def.layers {
            db.add_layer_to_collection(*layer, def.id).await?;
        }

        Ok(())
    }

    let dir = fs::read_dir(file_path);
    if dir.is_err() {
        warn!("Skipped adding layer collections from directory because it can't be read");
        return;
    }
    let dir = dir.expect("checked");

    let mut collection_defs = vec![];

    for entry in dir {
        match entry {
            Ok(entry) if entry.path().extension() == Some(OsStr::new("json")) => {
                match get_layer_collection_from_dir_entry(&entry) {
                    Ok(def) => collection_defs.push(def),
                    Err(e) => {
                        warn!(
                            "Skipped adding layer collection from directory entry: {:?} error: {}",
                            entry,
                            e.to_string()
                        );
                    }
                }
            }
            _ => {
                warn!(
                    "Skipped adding layer collection from directory entry: {:?}",
                    entry
                );
            }
        }
    }

    let mut collection_children: HashMap<LayerCollectionId, Vec<LayerCollectionId>> =
        HashMap::new();

    for def in collection_defs {
        let collection = add_collection_to_db(db, &def).await;

        match collection {
            Ok(_) => {
                collection_children.insert(def.id, def.collections);
            }
            Err(e) => {
                warn!("Skipped adding layer collection to db: {}", e);
            }
        }
    }

    for (parent, children) in collection_children {
        for child in children {
            let op = db.add_collection_to_parent(child, parent).await;

            if let Err(e) = op {
                warn!("Skipped adding child collection to db: {}", e);
            }
        }
    }
}
