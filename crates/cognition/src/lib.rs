mod config;
mod engine;
pub mod models;
pub mod tools;

pub use engine::{run_decision, Decision, DecisionPromptTemplate, DecisionResult, DecisionState};

#[derive(Debug)]
pub struct CognitionError(pub String);

impl std::fmt::Display for CognitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cognition error: {}", self.0)
    }
}
