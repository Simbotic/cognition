use cognition::{run_decision, CognitionError, Decision, DecisionPromptTemplate, DecisionState};
use std::fs::File;
use std::io::{Read, Write};

#[tokio::main]
async fn main() -> Result<(), CognitionError> {
    let decision_prompt_template = {
        let mut file = File::open("decision_prompt_template.yaml").unwrap();
        let mut decision_prompt_template = String::new();
        file.read_to_string(&mut decision_prompt_template).unwrap();
        DecisionPromptTemplate::new(decision_prompt_template)
    };

    let decision_nodes = {
        // Load the YAML file containing decision nodes
        let mut file = File::open("decision_tree.yaml").unwrap();
        let mut decision_nodes = String::new();
        file.read_to_string(&mut decision_nodes).unwrap();

        let decision_nodes: Vec<Decision> = serde_yaml::from_str(&decision_nodes).unwrap();
        decision_nodes
    };

    let config = format!(
        r#"
    models:
      davinci003:
        api_key: {}
    tools:
      wolfram_alpha:
        api_key: {}
    "#,
        std::env::var("OPENAI_API_KEY").unwrap(),
        std::env::var("WOLFRAM_APP_ID").unwrap()
    );

    let mut state = DecisionState::new(&config, decision_prompt_template, decision_nodes);

    let mut user_input = None;
    while let Some(result) = run_decision(user_input, &mut state).await? {
        // Print decision prompt, if any
        if let Some(decision_prompt) = result.decision_prompt {
            println!("\n++++++ PROMPT ++++++");
            println!("{}", decision_prompt);
            println!("--------------------");
        }

        // Print choice if any
        if let Some(choice) = result.choice {
            println!("\nCHOICE: {}", choice);
        }

        // Print tool results, if any
        if let Some(tool_response) = result.tool_response {
            println!("\nTOOL: [{}] {}", tool_response.id, tool_response.response);
        }

        // Display the current decision text and choices
        println!("\n>>>> DECISION: {}", result.decision_node.id);
        println!("\n{}: {}", state.agent, result.decision_node.text);
        for choice in &result.decision_node.choices {
            println!("- {}", choice.choice);
        }

        // Get user input
        let mut input = String::new();
        print!("{}: ", state.user);
        std::io::stdout().flush().unwrap();
        std::io::stdin().read_line(&mut input).unwrap();
        user_input = Some(input.trim().to_string());
    }

    Ok(())
}
