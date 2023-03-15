use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_yaml;
use std::collections::HashMap;
use std::fs::File;
use std::io::{
    Write, {BufReader, Read},
};
use tokio::runtime::Runtime;

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

// Send an asynchronous request to the OpenAI API
async fn send_request(prompt: &str) -> Result<OpenAIResponse, Box<dyn std::error::Error>> {
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
    let response_json: OpenAIResponse = serde_json::from_str(&response_text)?;
    Ok(response_json)
}

// Run the decision-making process using the decision tree
async fn run_decision() -> Result<(), Box<dyn std::error::Error>> {
    // Load the YAML file containing decision nodes
    let file = File::open("decision_tree.yaml")?;
    let reader = BufReader::new(file);
    let decision_nodes: Vec<Decision> = serde_yaml::from_reader(reader)?;

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

    let agent = "Agent";
    let user = "User";

    let mut history = String::new();

    // Initialize the decision loop
    let mut current_id = "start".to_string();
    let mut predicting_choice = false;
    let mut user_response = String::new();

    loop {
        // Find the current decision node
        let decision_node = decision_nodes
            .iter()
            .find(|obj| obj.id == current_id)
            .ok_or("Oops, something went wrong. Please try again.")?;

        println!("\n>>>> DECISION: {}\n", decision_node.id);

        if decision_node.id == "start" {
            // If the user chooses to start over, reset the decision loop
            history = String::new();
        }

        // If node has a tool, run the tool
        if let Some(tool) = &decision_node.tool {
            // Find the tool
            let tool = tools
                .iter()
                .find(|obj| obj.id == *tool)
                .ok_or("Oops, something went wrong. Please try again.")?;
            let client = reqwest::Client::new();
            let headers = HeaderMap::new();

            // Create params for tool
            let mut params = tool.params.clone();
            params.insert("i".to_string(), user_response.clone());

            // Create query string from params
            let query_string = serde_urlencoded::to_string(params).unwrap();
            let url = format!("{}?{}", tool.endpoint, query_string);

            // Send request to AI tool
            let response = client.get(&url).headers(headers).send().await?;
            println!("{}: {}", agent, response.text().await?);
        }

        // If node doesn't support prediction, disable prediction
        if let Some(false) = decision_node.predict {
            predicting_choice = false;
        }

        // Display the current text and choices
        println!("{}: {}", agent, decision_node.text);
        for choice in &decision_node.choices {
            println!("- {}", choice.choice);
        }

        // Update the history with the current text
        if !predicting_choice {
            if history.len() > 0 {
                history.push_str(&format!("\n  {}: {}", agent, decision_node.text));
            } else {
                history.push_str(&format!("{}: {}", agent, decision_node.text));
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
        if predicting_choice {
            print!("Predicting choice...");
        } else {
            let mut user_input = String::new();
            print!("{}: ", user);
            std::io::stdout().flush()?;
            std::io::stdin().read_line(&mut user_input)?;
            user_response = user_input.trim().to_string();

            // Update the history with the user's response
            history.push_str(&format!("\n  {}: {}", user, user_response));
        }

        // Create the decision prompt
        let decision_prompt = decision_node.text.clone();
        let decision_prompt =
            decision_prompt_template.format(&history, &decision_prompt, &choices, &user_response);

        println!("\n++++++ PROMPT ++++++");
        print!("{}", decision_prompt);

        // Send the request to OpenAI asynchronously
        let response = send_request(&decision_prompt).await?;

        // Get the first choice from the response
        let choice = response
            .choices
            .get(0)
            .ok_or("OpenAI did not return any choices")?
            .text
            .trim()
            .to_string();

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
                    println!("{}: Thank you for using the cognition system.", agent);
                    break;
                } else if next_id == "start" {
                    current_id = next_id;
                    // Disable prediction if we are restarting the decision loop
                    predicting_choice = false;
                } else {
                    // Otherwise, continue to the next decision node
                    current_id = next_id;
                    // Try to predict the user's next choice
                    predicting_choice = true;
                }
            }
            None => {
                if predicting_choice {
                    // If user's choice could not be predicted, disable prediction and repeat the prompt
                    println!("Failed to predict the user's choice.");
                } else {
                    // If no match is found, repeat the prompt
                    println!("{}: I'm sorry, I didn't understand your response.", agent);
                }

                // Disable prediction
                predicting_choice = false;
            }
        }
    }

    Ok(())
}

fn main() {
    let rt = Runtime::new().unwrap();
    rt.block_on(run_decision()).unwrap();
}
