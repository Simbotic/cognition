use crate::models::{InferenceResult, LargeLanguageModel, ModelError};
use async_trait::async_trait;
use reqwest::{
    header::{HeaderMap, HeaderValue, CONTENT_TYPE},
    Client, StatusCode,
};
use serde::Deserialize;
use serde_json::{json, Value};

pub struct Textgen {
    server: String,
    client: Client,
}

// Generation parameters
// Reference: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
#[derive(Debug, PartialEq, Clone)]
pub struct TextgenParams {
    pub max_new_tokens: usize,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub typical_p: f32,
    pub repetition_penalty: f32,
    pub encoder_repetition_penalty: f32,
    pub top_k: usize,
    pub min_length: usize,
    pub no_repeat_ngram_size: usize,
    pub num_beams: usize,
    pub penalty_alpha: f32,
    pub length_penalty: f32,
    pub early_stopping: bool,
}

impl TextgenParams {
    fn to_json_data(&self, prompt: &str) -> Value {
        json!({
            "data": [
                prompt,
                self.max_new_tokens,
                self.do_sample,
                self.temperature,
                self.top_p,
                self.typical_p,
                self.repetition_penalty,
                self.encoder_repetition_penalty,
                self.top_k,
                self.min_length,
                self.no_repeat_ngram_size,
                self.num_beams,
                self.penalty_alpha,
                self.length_penalty,
                self.early_stopping,
            ]
        })
    }
}

#[derive(Debug, Deserialize)]
pub struct TextgenResponse {
    data: Vec<Option<String>>,
    pub is_generating: bool,
    pub duration: f64,
    pub average_duration: f64,
}

#[async_trait(?Send)]
impl LargeLanguageModel for Textgen {
    fn new(_config: &str) -> Result<Self, ModelError> {
        Ok(Textgen {
            server: std::env::var("TEXTGEN_SERVER").map_err(|e| {
                ModelError::new(&format!("Cannot get TEXTGEN_SERVER from env var: {}", e))
            })?,
            client: Client::new(),
        })
    }

    async fn generate(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f32,
    ) -> Result<InferenceResult, ModelError> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let params = TextgenParams {
            max_new_tokens: max_length,
            do_sample: true,
            temperature,
            top_p: 0.9,
            typical_p: 1.0,
            repetition_penalty: 1.05,
            encoder_repetition_penalty: 1.0,
            top_k: 0,
            min_length: 0,
            no_repeat_ngram_size: 0,
            num_beams: 1,
            penalty_alpha: 0.0,
            length_penalty: 1.0,
            early_stopping: true,
        };

        let request_body = params.to_json_data(prompt);
        let response = self
            .client
            .post(format!("{}/run/textgen", self.server))
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ModelError::new(&format!("HTTP request error: {}", e)))?;

        let status = response.status();

        if status != StatusCode::OK {
            let error_body = response
                .text()
                .await
                .unwrap_or_else(|_| String::from("No error details"));
            println!("Error body: {}", error_body); // Print the response body with the error message
            return Err(ModelError::new(&format!(
                "Error {}: {}",
                status, error_body
            )));
        }

        let response_data = response
            .json::<TextgenResponse>()
            .await
            .map_err(|e| ModelError::new(&format!("JSON parsing error: {}", e)))?;
        let result = InferenceResult {
            text: response_data.data[0]
                .clone()
                .unwrap_or_else(|| String::from("No data found")),
            probabilities: vec![], // You may want to calculate probabilities based on your requirements
        };

        Ok(result)
    }
}
