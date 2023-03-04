FROM rust:1.67.1-alpine
WORKDIR parachain
COPY specs specs
COPY scripts scripts
COPY node node
COPY pallets pallets
COPY runtime runtime
COPY Cargo.toml Cargo.lock .
RUN cargo build --release
ENTRYPOINT scripts/run-collator.sh
