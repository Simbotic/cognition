use async_trait::async_trait;

pub mod openai;
pub mod davinci003;

#[derive(Debug)]
pub struct InferenceResult {
    pub text: String,
    pub probabilities: Vec<f32>,
}

#[async_trait]
pub trait LargeLanguageModel {
    /// Initializes the model with the given configuration.
    fn new(config: &str) -> Result<Self, Box<dyn std::error::Error>>
    where
        Self: Sized;

    /// Generates a response based on the given prompt.
    async fn generate_response(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f32,
    ) -> Result<InferenceResult, Box<dyn std::error::Error>>;

    /// Completes the given text with the most probable sequence.
    async fn complete_text(
        &self,
        text: &str,
        max_length: usize,
    ) -> Result<InferenceResult, Box<dyn std::error::Error>>;
}
