use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

// OpenAI API response
#[derive(Serialize, Deserialize)]
struct OpenAIResponse {
    id: String,
    object: String,
    created: usize,
    model: String,
    choices: Vec<OpenAIChoice>,
}

// A choice in OpenAI API response
#[derive(Serialize, Deserialize)]
struct OpenAIChoice {
    text: String,
    index: usize,
    logprobs: Option<OpenAILogprobs>,
    finish_reason: String,
}

// Log probabilities in OpenAI API response
#[derive(Serialize, Deserialize)]
struct OpenAILogprobs {
    top_logprobs: HashMap<String, f64>,
    text_offset: Vec<usize>,
}

// Send an asynchronous request to the OpenAI API
pub async fn send_request(prompt: &str) -> Result<String, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut headers = HeaderMap::new();
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    headers.insert(
        AUTHORIZATION,
        HeaderValue::from_str(&format!(
            "Bearer {}",
            std::env::var("OPENAI_API_KEY").unwrap()
        ))?,
    );
    let response = client
        .post("https://api.openai.com/v1/completions")
        .headers(headers)
        .json(&json!({
            "model": "text-davinci-003",
            "prompt": prompt,
            "suffix": "\n\n",
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }))
        .send()
        .await?;
    let response_text = response.text().await?;
    let response: OpenAIResponse = serde_json::from_str(&response_text)?;
    let response = response
        .choices
        .get(0)
        .ok_or("OpenAI did not return any choices")?
        .text
        .trim()
        .to_string();
    Ok(response)
}
