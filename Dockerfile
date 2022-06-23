FROM buildpack-deps:bookworm as rust-base

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH

RUN set -eux; \
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
        amd64) rustArch='x86_64-unknown-linux-gnu' ;; \
        arm64) rustArch='aarch64-unknown-linux-gnu' ;; \
        *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
    \
    url="https://static.rust-lang.org/rustup/dist/${rustArch}/rustup-init"; \
    wget "$url"; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --default-toolchain nightly; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;


RUN apt update && apt install -y --no-install-recommends \
    clang mold libgcc-10-dev gdal-bin libgdal-dev cmake \
    && rm -rf /var/lib/apt/lists/*


RUN cargo install cargo-chef



#################################################################################
# PLANNER
#################################################################################

FROM rust-base as planner

WORKDIR app


COPY . .

# Compute a recipe file out of Cargo.toml
RUN cargo chef prepare --recipe-path recipe.json

#################################################################################
# CACHER
#################################################################################

FROM rust-base as cacher
     
WORKDIR app

# Get the recipe file
COPY --from=planner /app/recipe.json recipe.json

# Cache dependencies
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo chef cook --tests --recipe-path recipe.json

#################################################################################
# BUILDER
#################################################################################

FROM rust-base as builder

WORKDIR app

COPY . .
# Copy built dependencies over cache

COPY --from=cacher /app/target target
# Copy cargo folder from cache. This includes the package registry and downloaded sources

COPY --from=cacher $CARGO_HOME $CARGO_HOME
# Build the binary

RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo build --tests

# #################################################################################
# # RUNTIME
# #################################################################################
   
FROM rust-base as runtime

WORKDIR app
COPY --from=builder /app/target target
COPY . .
#
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    --mount=type=cache,target=/app/target \
    cargo test xg_reservoir_test -- --nocapture
#
#  
#
#
