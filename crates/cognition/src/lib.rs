mod engine;
pub mod models;
pub mod tools;

pub use engine::{DecisionState, run_decision, DecisionResult};

#[derive(Debug)]
pub struct CognitionError(String);

impl std::fmt::Display for CognitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cognition error: {}", self.0)
    }
}
