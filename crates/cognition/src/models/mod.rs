use async_trait::async_trait;

pub mod davinci003;
pub mod textgen;

use std::error::Error;
use std::fmt::{self, Display};

#[derive(Debug)]
pub struct ModelError {
    message: String,
}

impl ModelError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_owned(),
        }
    }
}

impl Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for ModelError {}

#[derive(Debug)]
pub struct InferenceResult {
    pub text: String,
    pub probabilities: Vec<f32>,
}

#[async_trait(?Send)]
pub trait LargeLanguageModel {
    /// Initializes the model with the given configuration.
    fn new(config: &String) -> Result<Self, ModelError>
    where
        Self: Sized;

    /// Generates a response based on the given prompt.
    async fn generate(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f32,
    ) -> Result<InferenceResult, ModelError>;
}
