use cognition::{run_decision, CognitionError, DecisionState};
use std::io::Write;

#[tokio::main]
async fn main() -> Result<(), CognitionError> {
    let mut state = DecisionState::default();

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
