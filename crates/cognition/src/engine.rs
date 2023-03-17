use crate::{
    models::{self, LargeLanguageModel},
    CognitionError,
};
use reqwest::header::HeaderMap;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;
use std::fs::File;
use std::io::{
    Write, {BufReader, Read},
};

// YAML decision node structure
#[derive(Serialize, Deserialize)]
struct Decision {
    id: String,
    text: String,
    tool: Option<String>,
    predict: Option<bool>,
    choices: Vec<Choice>,
}

// Choice structure within a decision node
#[derive(Serialize, Deserialize)]
struct Choice {
    choice: String,
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
        user_response: &str,
    ) -> String {
        self.0
            .replace("{{history}}", history)
            .replace("{{decision_prompt}}", decision_prompt)
            .replace("{{choices}}", choices)
            .replace("{{user_response}}", user_response)
    }
}

pub struct DecisionState {
    model: Box<dyn LargeLanguageModel>,
    decision_nodes: Vec<Decision>,
    decision_prompt_template: DecisionPromptTemplate,
    tools: Vec<Tool>,
    agent: String,
    user: String,
    history: String,
    choices: String,
    user_response: String,
    current_id: String,
    predicting_choice: bool,
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
        let predicting_choice = false;
        let user_response = String::new();

        Self {
            model: Box::new(model),
            decision_nodes,
            decision_prompt_template,
            tools,
            agent,
            user,
            history,
            choices: String::new(),
            user_response,
            current_id,
            predicting_choice,
        }
    }
}

#[derive(Debug)]
pub struct DecisionResult {
    pub prev_id: String,
    pub choice: String,
    pub decision_prompt: String,
    pub tool_response: Option<String>,
}

// Run the decision-making process using the decision tree
pub async fn run_decision(
    state: &mut DecisionState,
) -> Result<Option<DecisionResult>, CognitionError> {
    // Find the current decision node
    let decision_node = state
        .decision_nodes
        .iter()
        .find(|obj| obj.id == state.current_id)
        .ok_or_else(|| {
            CognitionError(format!(
                "Could not find decision node: {}",
                state.current_id
            ))
        })?;

    println!("\n>>>> DECISION: {}\n", decision_node.id);

    if decision_node.id == "start" {
        // If the user chooses to start over, reset the decision loop
        state.history = String::new();
    }

    let mut tool_response = None;

    // If node has a tool, run the tool
    if let Some(tool) = &decision_node.tool {
        // Find the tool
        let tool = state
            .tools
            .iter()
            .find(|obj| obj.id == *tool)
            .ok_or_else(|| CognitionError(format!("Could not find tool: {}", tool)))?;
        let client = reqwest::Client::new();
        let headers = HeaderMap::new();

        // Create params for tool
        let mut params = tool.params.clone();
        params.insert("i".to_string(), state.user_response.clone());

        // Create query string from params
        let query_string = serde_urlencoded::to_string(params).unwrap();
        let url = format!("{}?{}", tool.endpoint, query_string);

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
        println!("{}: {}", state.agent, response);
        tool_response = Some(response);
    }

    // If node doesn't support prediction, disable prediction
    if let Some(false) = decision_node.predict {
        state.predicting_choice = false;
    }

    // Display the current text and choices
    println!("{}: {}", state.agent, decision_node.text);
    for choice in &decision_node.choices {
        println!("- {}", choice.choice);
    }

    // Update the history with the current text
    if !state.predicting_choice {
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

    // Map choices to choices.choice
    let choices: Vec<String> = decision_node
        .choices
        .iter()
        .map(|choice| choice.choice.trim().to_string())
        .collect();
    let choices = choices.join("\n  - ");

    // Prompt the user for input unless predicting the choice
    if state.predicting_choice {
        print!("Predicting choice...");
    } else {
        let mut user_input = String::new();
        print!("{}: ", state.user);
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut user_input).unwrap();
        state.user_response = user_input.trim().to_string();

        // Update the history with the user's response
        state
            .history
            .push_str(&format!("\n  {}: {}", state.user, state.user_response));
    }

    // Create the decision prompt
    let decision_prompt = decision_node.text.clone();
    let decision_prompt = state.decision_prompt_template.format(
        &state.history,
        &decision_prompt,
        &choices,
        &state.user_response,
    );

    println!("\n++++++ PROMPT ++++++");
    print!("{}", decision_prompt);

    // Send the request to OpenAI asynchronously
    let choice = state
        .model
        .generate(&decision_prompt, 200, 0.5)
        .await
        .map_err(|err| CognitionError(format!("Failed to generate choice: {}", err)))?;
    let choice = choice.text;
    println!("{}", choice);
    println!("--------------------");

    // Try to match the user's response with one of the choices
    let choice_index = decision_node
        .choices
        .iter()
        .position(|o| o.choice == choice);

    match choice_index {
        Some(index) => {
            // Get the next decision node ID based on the user's choice
            let next_id = decision_node.choices[index].next_id.clone();

            if next_id == "exit" {
                // If the user chooses to exit, end the decision loop
                println!("{}: Thank you for using the cognition system.", state.agent);
                return Ok(None);
            } else if next_id == "start" {
                state.current_id = next_id;
                // Disable prediction if we are restarting the decision loop
                state.predicting_choice = false;
            } else {
                // Otherwise, continue to the next decision node
                state.current_id = next_id;
                // Try to predict the user's next choice
                state.predicting_choice = true;
            }
        }
        None => {
            if state.predicting_choice {
                // If user's choice could not be predicted, disable prediction and repeat the prompt
                println!("Failed to predict the user's choice.");
            } else {
                // If no match is found, repeat the prompt
                println!(
                    "{}: I'm sorry, I didn't understand your response.",
                    state.agent
                );
            }

            // Disable prediction
            state.predicting_choice = false;
        }
    }

    let result = DecisionResult {
        tool_response,
        choice,
        decision_prompt,
        prev_id: decision_node.id.clone(),
    };

    Ok(Some(result))
}
