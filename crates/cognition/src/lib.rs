pub mod engine;
pub mod models;
pub mod tools;

#[derive(Debug)]
pub struct CognitionError(String);

impl std::fmt::Display for CognitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cognition error: {}", self.0)
    }
}