use crate::models::{InferenceResult, LargeLanguageModel};
use async_trait::async_trait;
use reqwest::{
    header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE},
    Client,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;

pub struct Davinci003 {
    client: Client,
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

#[async_trait]
impl LargeLanguageModel for Davinci003 {
    fn new(_config: &str) -> Result<Self, Box<dyn Error>> {
        let client = Client::new();
        Ok(Self { client })
    }

    async fn generate_response(
        &self,
        prompt: &str,
        max_length: usize,
        temperature: f32,
    ) -> Result<InferenceResult, Box<dyn Error>> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!(
                "Bearer {}",
                std::env::var("OPENAI_API_KEY").unwrap()
            ))?,
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
            .await?
            .json::<OpenAIResponse>()
            .await?;

        let choice = response.choices.get(0).ok_or("No choices found")?;
        let result = InferenceResult {
            text: choice.text.clone(),
            probabilities: vec![], // You may want to calculate probabilities based on your requirements
        };

        Ok(result)
    }

    async fn complete_text(
        &self,
        text: &str,
        max_length: usize,
    ) -> Result<InferenceResult, Box<dyn Error>> {
        self.generate_response(text, max_length, 1.0).await
    }
}
