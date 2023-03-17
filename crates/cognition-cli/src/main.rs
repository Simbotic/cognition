use cognition::{engine, CognitionError};

#[tokio::main]
async fn main() -> Result<(), CognitionError> {
    engine::run_decision().await
}
