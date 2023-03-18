use crate::{
    config::string_by_path,
    models::{InferenceResult, LargeLanguageModel, ModelError},
};
use async_trait::async_trait;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub struct Davinci003 {
    client: Client,
    api_key: String,
}

#[derive(Serialize)]
struct OpenAIRequestBody<'a> {
    model: &'a str,
    prompt: &'a str,
    suffix: &'a str,
    temperature: f32,
    max_tokens: usize,
    top_p: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
}

#[derive(Serialize, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: usize,
    model: String,
    choices: Vec<OpenAIChoice>,
}

#[derive(Serialize, Deserialize)]
struct OpenAIChoice {
    text: String,
    index: usize,
    logprobs: Option<OpenAILogprobs>,
    finish_reason: String,
}

#[derive(Serialize, Deserialize)]
struct OpenAILogprobs {
    top_logprobs: HashMap<String, f64>,
    text_offset: Vec<usize>,
}

#[async_trait(?Send)]
impl LargeLanguageModel for Davinci003 {
    fn new(config: &String) -> Result<Self, ModelError> {
        let client = Client::new();
        let api_key = string_by_path(config, "models.davinci003.api_key").unwrap();
        Ok(Self {
            client,
            api_key: api_key.to_string(),
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
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", self.api_key,))
                .map_err(|e| ModelError::new(&format!("Authorization header error: {}", e)))?,
        );

        let request_body = OpenAIRequestBody {
            model: "text-davinci-003",
            prompt,
            suffix: "\n\n",
            temperature,
            max_tokens: max_length,
            top_p: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/completions")
            .headers(headers)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| ModelError::new(&format!("HTTP request error: {}", e)))?
            .json::<OpenAIResponse>()
            .await
            .map_err(|e| ModelError::new(&format!("JSON parsing error: {}", e)))?;

        let choice = response
            .choices
            .get(0)
            .ok_or_else(|| ModelError::new("No choices found"))?;
        let result = InferenceResult {
            text: choice.text.clone(),
            probabilities: vec![], // You may want to calculate probabilities based on your requirements
        };

        Ok(result)
    }
}
