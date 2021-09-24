use flexi_logger::{
    style, AdaptiveFormat, Age, Cleanup, Criterion, DeferredNow, Duplicate, FileSpec, Logger,
    LoggerHandle, Naming, WriteMode,
};
use geoengine_services::error::{Error, Result};
use geoengine_services::util::config;
use geoengine_services::util::config::get_config_element;
use log::Record;

#[actix_web::main]
async fn main() -> Result<(), Error> {
    let logger = initialize_logging()?;

    let res = start_server().await;

    logger.shutdown();
    res
}

#[cfg(not(feature = "pro"))]
pub async fn start_server() -> Result<()> {
    geoengine_services::server::start_server(None).await
}

#[cfg(feature = "pro")]
pub async fn start_server() -> Result<()> {
    geoengine_services::pro::server::start_pro_server(None).await
}

fn initialize_logging() -> Result<LoggerHandle> {
    let logging_config: config::Logging = get_config_element()?;

    let mut logger = Logger::try_with_str(logging_config.log_spec)?
        .format(custom_log_format)
        .adaptive_format_for_stderr(AdaptiveFormat::Custom(
            custom_log_format,
            colored_custom_log_format,
        ));

    if logging_config.log_to_file {
        let mut file_spec = FileSpec::default().basename(logging_config.filename_prefix);

        if let Some(dir) = logging_config.log_directory {
            file_spec = file_spec.directory(dir);
        }

        logger = logger
            .log_to_file(file_spec)
            .write_mode(if logging_config.enable_buffering {
                WriteMode::BufferAndFlush
            } else {
                WriteMode::Direct
            })
            .append()
            .rotate(
                Criterion::Age(Age::Day),
                Naming::Timestamps,
                Cleanup::KeepLogFiles(7),
            )
            .duplicate_to_stderr(Duplicate::All);
    }

    Ok(logger.start()?)
}

/// A logline-formatter that produces log lines like
/// <br>
/// ```[2021-05-18 10:16:53 +02:00] INFO [my_prog::some_submodule] Task successfully read from conf.json```
/// <br>
fn custom_log_format(
    w: &mut dyn std::io::Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    write!(
        w,
        "[{}] {} [{}] {}",
        now.now().format("%Y-%m-%d %H:%M:%S %:z"),
        record.level(),
        record.module_path().unwrap_or("<unnamed>"),
        &record.args()
    )
}

/// A colored version of the logline-formatter `custom_log_format`.
fn colored_custom_log_format(
    w: &mut dyn std::io::Write,
    now: &mut DeferredNow,
    record: &Record,
) -> Result<(), std::io::Error> {
    let level = record.level();
    let style = style(level);
    write!(
        w,
        "[{}] {} [{}] {}",
        style.paint(now.now().format("%Y-%m-%d %H:%M:%S %:z").to_string()),
        style.paint(level.to_string()),
        record.module_path().unwrap_or("<unnamed>"),
        style.paint(&record.args().to_string())
    )
}
