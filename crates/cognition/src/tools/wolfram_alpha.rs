use super::*;

pub struct WolframAlpha {
    pub id: String,
    pub name: String,
    pub description: String,
    pub endpoint: Url,
    pub params: HashMap<String, String>,
}

impl WolframAlpha {
    pub fn new(app_id: String) -> Self {
        Self {
            id: "wolfram_alpha".to_string(),
            name: "Wolfram|Alpha".to_string(),
            description: "Wolfram Alpha is a computational knowledge engine".to_string(),
            endpoint: "https://api.wolframalpha.com/v1/result".try_into().unwrap(),
            params: vec![("appid".to_string(), app_id)].into_iter().collect(),
        }
    }
}

#[async_trait(?Send)]
impl Tool for WolframAlpha {
    fn id(&self) -> &String {
        &self.id
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn description(&self) -> &String {
        &self.description
    }

    async fn run(&self, input: &String) -> Result<Option<ToolResponse>, CognitionError> {
        let client = reqwest::Client::new();
        let headers = HeaderMap::new();

        let mut params = self.params.clone();
        params.insert("i".to_string(), input.clone());

        // Create query string from params
        let query_string = serde_urlencoded::to_string(params).unwrap();
        let url = format!("{}?{}", self.endpoint, query_string);

        // Send request to AI tool
        let response = client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .map_err(|err| CognitionError(format!("Failed to send request to tool: {}", err)))?;

        let response = response
            .text()
            .await
            .map_err(|err| CognitionError(format!("Failed to get response text: {}", err)))?;
        debug!("{}: {}", self.id, response);
        Ok(Some(ToolResponse {
            id: self.id.clone(),
            response: response,
        }))
    }
}
