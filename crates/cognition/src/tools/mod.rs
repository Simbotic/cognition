use crate::CognitionError;
use async_trait::async_trait;
use log::debug;
use reqwest::{header::HeaderMap, Url};
use std::collections::HashMap;

// Easy access to tools
pub use signal::Signal;
pub use wolfram_alpha::WolframAlpha;

mod signal;
mod wolfram_alpha;

#[async_trait(?Send)]
pub trait Tool {
    fn id(&self) -> &String;
    fn name(&self) -> &String;
    fn description(&self) -> &String;
    async fn run(&self, input: &String) -> Result<Option<ToolResponse>, CognitionError>;
}

#[derive(Debug, Clone)]
pub struct ToolResponse {
    pub id: String,
    pub response: String,
}
