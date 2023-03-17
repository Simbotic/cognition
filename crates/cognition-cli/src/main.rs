use cognition::{run_decision, CognitionError, DecisionState};

#[tokio::main]
async fn main() -> Result<(), CognitionError> {
    let mut state = DecisionState::default();

    while let Some(result) = run_decision(&mut state).await? {
        println!("{:?}", result);
    }

    Ok(())
}
