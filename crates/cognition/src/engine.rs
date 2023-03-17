use crate::{
    models::{self, LargeLanguageModel},
    CognitionError,
};
use log::debug;
use reqwest::header::HeaderMap;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};

// YAML decision node structure
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Decision {
    pub id: String,
    pub text: String,
    pub tool: Option<String>,
    pub predict: Option<bool>,
    pub choices: Vec<Choice>,
}

// Choice structure within a decision node
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Choice {
    pub choice: String,
    next_id: String,
}

#[derive(Serialize, Deserialize)]
struct Tool {
    id: String,
    name: String,
    description: String,
    endpoint: Url,
    params: HashMap<String, String>,
}

// YAML prompt_decision template object
struct DecisionPromptTemplate(String);

impl DecisionPromptTemplate {
    fn new(file_path: &str) -> Self {
        let mut file = File::open(file_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        Self(contents)
    }

    // Format the decision prompt template with the given parameters
    fn format(
        &self,
        history: &str,
        decision_prompt: &str,
        choices: &str,
        user_input: &str,
    ) -> String {
        self.0
            .replace("{{history}}", history)
            .replace("{{decision_prompt}}", decision_prompt)
            .replace("{{choices}}", choices)
            .replace("{{user_input}}", user_input)
    }
}

pub struct DecisionState {
    model: Box<dyn LargeLanguageModel>,
    decision_nodes: Vec<Decision>,
    decision_prompt_template: DecisionPromptTemplate,
    tools: Vec<Tool>,
    pub agent: String,
    pub user: String,
    history: String,
    current_id: String,
}

impl DecisionState {
    fn decision_node(&self, id: &str) -> Result<&Decision, CognitionError> {
        self.decision_nodes
            .iter()
            .find(|node| node.id == id)
            .ok_or_else(|| CognitionError(format!("Decision node with ID '{}' not found", id)))
    }

    pub fn current_node(&self) -> Result<&Decision, CognitionError> {
        self.decision_node(&self.current_id)
    }
}

impl Default for DecisionState {
    fn default() -> Self {
        // LLM model
        let model = models::davinci003::Davinci003::new("").unwrap();
        // let model = models::textgen::Textgen::new("").unwrap();

        // Load the YAML file containing decision nodes
        let file = File::open("decision_tree.yaml")
            .map_err(|err| CognitionError(format!("Failed to open decision tree file: {}", err)))
            .unwrap();
        let reader = BufReader::new(file);
        let decision_nodes: Vec<Decision> = serde_yaml::from_reader(reader)
            .map_err(|err| CognitionError(format!("Failed to parse decision tree YAML: {}", err)))
            .unwrap();

        // Load the decision prompt template from the YAML file
        let decision_prompt_template = DecisionPromptTemplate::new("decision_prompt_template.yaml");

        // Load all available AI tools
        let tools = vec![Tool {
            id: "wolfram_alpha".to_string(),
            name: "Wolfram|Alpha".to_string(),
            description: "AI tool for answering factual and mathematical questions.".to_string(),
            endpoint: "https://api.wolframalpha.com/v1/result".try_into().unwrap(),
            params: vec![(
                "appid".to_string(),
                std::env::var("WOLFRAM_APP_ID").unwrap(),
            )]
            .into_iter()
            .collect(),
        }];

        let agent = "Agent".into();
        let user = "User".into();

        let history = String::new();

        // Initialize the decision loop
        let current_id = "start".to_string();

        Self {
            model: Box::new(model),
            decision_nodes,
            decision_prompt_template,
            tools,
            agent,
            user,
            history,
            current_id,
        }
    }
}

#[derive(Debug)]
pub struct ToolResponse {
    pub id: String,
    pub response: String,
}

#[derive(Debug)]
pub struct DecisionResult {
    pub user_input: Option<String>,
    pub decision_prompt: Option<String>,
    pub choice: Option<String>,
    pub current_id: String,
    pub decision_node: Decision,
    pub tool_response: Option<ToolResponse>,
}

// Run the decision-making process using the decision tree
pub async fn run_decision(
    user_input: Option<String>,
    state: &mut DecisionState,
) -> Result<Option<DecisionResult>, CognitionError> {
    let mut predicting_choice = false;
    let mut tool_response = None;
    let mut decision_prompt = None;
    let mut choice;

    loop {
        choice = None;

        if let Some(user_input) = &user_input {
            // Update the history with the user's response
            if !predicting_choice {
                state
                    .history
                    .push_str(&format!("\n  {}: {}", state.user, user_input));
            }

            let decision_node = state.decision_node(&state.current_id)?.clone();

            // Map choices to choices.choice
            let choices: Vec<String> = decision_node
                .choices
                .iter()
                .map(|choice| choice.choice.trim().to_string())
                .collect();
            let choices = choices.join("\n  - ");

            // Create the decision prompt
            let prompt = decision_node.text.clone();
            let mut prompt = state.decision_prompt_template.format(
                &state.history,
                &prompt,
                &choices,
                &user_input,
            );

            // Send the request to OpenAI asynchronously
            let response = state
                .model
                .generate(&prompt, 200, 0.5)
                .await
                .map_err(|err| CognitionError(format!("Failed to generate choice: {}", err)))?;
            let response = response.text;
            prompt.push_str(&response);
            debug!("{}", prompt);

            // Set current prompt
            decision_prompt = Some(prompt);

            // Try to match the user's response with one of the choices
            let choice_index = decision_node
                .choices
                .iter()
                .position(|o| o.choice == response);

            match choice_index {
                Some(index) => {
                    // The user's response matches one of the choices, we have a choice
                    choice = Some(response.clone());
                    // Get the next decision node ID based on the user's choice
                    let next_id = decision_node.choices[index].next_id.clone();

                    if next_id == "exit" {
                        // If the user chooses to exit, end the decision loop
                        debug!("{}: Thank you for using the cognition system.", state.agent);
                        return Ok(None);
                    } else if next_id == "start" {
                        state.current_id = next_id;
                        // Disable prediction if we are restarting the decision loop
                        predicting_choice = false;
                    } else {
                        // Otherwise, continue to the next decision node
                        state.current_id = next_id;
                        // Try to predict the user's next choice
                        predicting_choice = true;
                    }
                }
                None => {
                    if predicting_choice {
                        // If user's choice could not be predicted, disable prediction and repeat the prompt
                        debug!("Failed to predict the user's choice.");
                    } else {
                        // If no match is found, repeat the prompt
                        debug!(
                            "{}: I'm sorry, I didn't understand your response.",
                            state.agent
                        );
                    }

                    // Disable prediction
                    predicting_choice = false;
                }
            }
        }

        // Find the current decision node
        let decision_node = state.decision_node(&state.current_id)?.clone();

        // If the user chooses to start over, reset the decision loop
        if decision_node.id == "start" {
            state.history = String::new();
        }

        // If node doesn't support prediction, disable prediction
        if let Some(false) = decision_node.predict {
            predicting_choice = false;
        }

        // Update the history with the current text
        if !predicting_choice {
            if state.history.len() > 0 {
                state
                    .history
                    .push_str(&format!("\n  {}: {}", state.agent, decision_node.text));
            } else {
                state
                    .history
                    .push_str(&format!("{}: {}", state.agent, decision_node.text));
            }
        }

        if let Some(user_input) = &user_input {
            // If node has a tool, run the tool
            if let Some(tool_id) = &decision_node.tool {
                // Find the tool
                let tool = state
                    .tools
                    .iter()
                    .find(|obj| obj.id == *tool_id)
                    .ok_or_else(|| CognitionError(format!("Could not find tool: {}", tool_id)))?;
                let client = reqwest::Client::new();
                let headers = HeaderMap::new();

                // Create params for tool
                let mut params = tool.params.clone();
                params.insert("i".to_string(), user_input.clone());

                // Create query string from params
                let query_string = serde_urlencoded::to_string(params).unwrap();
                let url = format!("{}?{}", tool.endpoint, query_string);

                // Send request to AI tool
                let response = client
                    .get(&url)
                    .headers(headers)
                    .send()
                    .await
                    .map_err(|err| {
                        CognitionError(format!("Failed to send request to tool: {}", err))
                    })?;

                let response = response.text().await.map_err(|err| {
                    CognitionError(format!("Failed to get response text: {}", err))
                })?;
                debug!("{}: {}", state.agent, response);
                tool_response = Some(ToolResponse {
                    id: tool_id.clone(),
                    response: response,
                });
            }
        }

        if !predicting_choice {
            break;
        }
    }

    let result = DecisionResult {
        user_input,
        decision_prompt,
        choice,
        current_id: state.current_id.clone(),
        decision_node: state.current_node()?.clone(),
        tool_response,
    };

    Ok(Some(result))
}
