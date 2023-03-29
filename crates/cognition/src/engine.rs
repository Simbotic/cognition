use crate::{
    models::{self, LargeLanguageModel},
    CognitionError, DecisionPromptTemplate, Tool, ToolResponse,
};
use log::*;
use serde::{Deserialize, Serialize};

// YAML decision node structure
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Decision {
    pub id: String,
    pub text: String,
    pub predicted_text: Option<String>,
    pub tool: Option<String>,
    pub predict: Option<bool>,
    pub reset: Option<bool>,
    pub choices: Option<Vec<Choice>>,
}

impl Decision {
    pub fn choices(&self) -> Vec<&Choice> {
        self.choices.iter().flat_map(|choice| choice).collect()
    }
}

// Choice structure within a decision node
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Choice {
    #[serde(rename = "choice")]
    pub text: String,
    next_id: String,
}

pub struct DecisionState {
    model: Box<dyn LargeLanguageModel>,
    decision_nodes: Vec<Decision>,
    decision_prompt_template: DecisionPromptTemplate,
    tools: Vec<Box<dyn Tool>>,
    pub agent: String,
    pub user: String,
    history: String,
    current_id: String,
}

impl DecisionState {
    pub fn new(
        config: &String,
        decision_prompt_template: DecisionPromptTemplate,
        decision_nodes: Vec<Decision>,
    ) -> Self {
        // LLM model
        let model = models::davinci003::Davinci003::new(config).unwrap();
        // let model = models::textgen::Textgen::new("").unwrap();

        let agent = "Agent".into();
        let user = "User".into();

        let history = String::new();

        // Initialize the decision loop
        let current_id = "start".to_string();

        Self {
            model: Box::new(model),
            decision_nodes,
            decision_prompt_template,
            tools: vec![],
            agent,
            user,
            history,
            current_id,
        }
    }

    // add tool
    pub fn add_tool(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

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

#[derive(Debug)]
pub struct DecisionResult {
    pub user_input: Option<String>,
    pub decision_prompt: Option<String>,
    pub choice: Option<String>,
    pub current_id: String,
    pub decision_node: Decision,
    pub predictions: Vec<Prediction>,
    pub tool_response: Option<ToolResponse>,
}

#[derive(Debug)]
pub struct Prediction {
    pub choice: String,
    pub id: String,
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
    let choice: Option<String> = None;
    let mut predictions = vec![];
    let mut max_depth = 5;

    loop {
        let decision_node = state.decision_node(&state.current_id)?.clone();

        // Map choices to choices.choice
        let choices: Vec<&Choice> = decision_node.choices();

        // If there are no choices, we're done
        if choices.len() == 0 {
            break;
        }

        // Select next choice
        let next_choice = if user_input.is_none() {
            // If user has not provided input, do not make a choice
            None
        } else if choices.len() == 1 {
            // If there is only one choice, select it
            debug!("Only one choice, skip prediction");
            choices.first()
        } else if let Some(user_input) = &user_input {
            // If many choices, predict best choice
            info!("User input: {:?}", user_input);

            // Map choices to choice string
            let choice_texts: Vec<String> = choices
                .iter()
                .map(|choice| choice.text.trim().to_string())
                .collect();

            let choices_str = choice_texts.join("\n  - ");

            // Create the decision prompt
            let prompt = decision_node.text.clone();
            let mut prompt = state.decision_prompt_template.format(
                &state.history,
                &prompt,
                &choices_str,
                &user_input,
            );

            // Few shot prediction
            let response = state
                .model
                .generate(&prompt, 200, 0.5)
                .await
                .map_err(|err| CognitionError(format!("Failed to generate choice: {}", err)))?;
            let response = response.text;
            prompt.push_str(&response);
            debug!("{}", &prompt);

            // Set current prompt
            decision_prompt = Some(prompt);

            // Try to match the user's response with one of the choices
            choice_texts
                .iter()
                .position(|choice| *choice == response)
                .and_then(|index| choices.get(index))
        } else {
            None
        };

        // Update the history with the agent-user interaction
        if let Some(user_input) = &user_input {
            if !predicting_choice {
                // Update the history with the current text
                if state.history.len() > 0 {
                    state.history.push_str(&format!("\n  "));
                }
                state
                    .history
                    .push_str(&format!("- {}: {}", state.agent, decision_node.text));
                // Update the history with the user's response
                state
                    .history
                    .push_str(&format!("\n  - {}: {}", state.user, user_input));
            }
        }

        // If there is a choice, get the next decision node ID
        if let Some(choice) = next_choice {
            info!(
                "Predicting the user's next choice... {} {}",
                decision_node.id, decision_node.text
            );
            predictions.push(Prediction {
                choice: choice.text.clone(),
                id: choice.next_id.clone(),
                tool_response: tool_response.clone(),
            });

            predicting_choice = true;
            // Continue to the next decision node
            state.current_id = choice.next_id.clone();
        }

        // Find the current decision node
        let decision_node = state.decision_node(&state.current_id)?.clone();

        // If node has reset, reset the history
        if let Some(true) = decision_node.reset {
            state.history = String::new();
        }

        // If node doesn't support prediction, disable prediction
        if let Some(false) = decision_node.predict {
            predicting_choice = false;
        }

        // If there is no choice, disable prediction
        if next_choice.is_none() {
            predicting_choice = false;
        }

        // If there is a tool, run the tool and get the response
        if let Some(user_input) = &user_input {
            // If node has a tool, run the tool
            if let Some(tool_id) = &decision_node.tool {
                // Find the tool
                let tool = state
                    .tools
                    .iter()
                    .find(|obj| *obj.id() == *tool_id)
                    .ok_or_else(|| CognitionError(format!("Could not find tool: {}", tool_id)))?;
                tool_response = tool.run(user_input).await?;
            }
        }

        max_depth -= 1;
        if !predicting_choice || max_depth == 0 {
            break;
        }
    }

    let result = DecisionResult {
        user_input,
        decision_prompt,
        choice,
        current_id: state.current_id.clone(),
        decision_node: state.current_node()?.clone(),
        predictions,
        tool_response,
    };

    Ok(Some(result))
}
