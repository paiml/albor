# Albor training environment
# Sovereign AI stack — zero external dependencies beyond Rust toolchain
FROM rust:1.89-bookworm AS builder

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /albor
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY tests/ tests/
COPY benches/ benches/
COPY contracts/ contracts/
COPY configs/ configs/

# Install sovereign tools
RUN cargo install bashrs pmat \
    && cargo install --git https://github.com/paiml/provable-contracts provable-contracts-cli

# Build and test
RUN cargo build --release && cargo test

FROM rust:1.89-slim-bookworm
WORKDIR /albor
COPY --from=builder /albor/ .
COPY --from=builder /usr/local/cargo/bin/pv /usr/local/cargo/bin/bashrs /usr/local/cargo/bin/pmat /usr/local/cargo/bin/
COPY docs/ docs/
COPY scripts/ scripts/
COPY models/ models/

ENTRYPOINT ["cargo", "test"]
