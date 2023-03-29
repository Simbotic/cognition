mod config;
mod engine;
mod models;
mod templates;
pub mod tools;

pub use engine::{run_decision, Decision, DecisionResult, DecisionState};
pub use templates::decision::DecisionPromptTemplate;
pub use tools::{Tool, ToolResponse};

#[derive(Debug)]
pub struct CognitionError(pub String);

impl std::fmt::Display for CognitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cognition error: {}", self.0)
    }
}
