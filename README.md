# Cognition System

The Cognition System is a decision-making tool that utilizes OpenAI's GPT-3 to interact with users and help them make decisions based on a predefined decision tree. The system also integrates with other AI tools like Wolfram|Alpha to answer factual and mathematical questions.

## Features

- Predefined decision tree in YAML format
- Integration with OpenAI's GPT-3 for decision-making
- Recursive prediction of user responses
- Integration with Wolfram|Alpha for answering factual and mathematical questions

## Installation

To install and run the Cognition System, you'll need Rust installed on your machine. If you don't have Rust installed, you can follow the instructions at https://www.rust-lang.org/tools/install.

### Clone the repository

```
git clone https://github.com/yourusername/cognition-system.git
cd cognition-system
```

### Configure API keys

You'll need to set environment variables for the OpenAI and Wolfram|Alpha API keys.
```
export OPENAI_API_KEY="your_openai_api_key"
export WOLFRAM_APP_ID="your_wolfram_app_id"
```

### Build and run

```
cargo run --release
```

## Usage

Once you've built and run the Cognition System, you'll be prompted with a series of questions and choices. You can navigate the decision tree by typing your choice and pressing Enter. To exit the system, type "exit" when prompted.

## Customization

To customize the decision tree, modify the `decision_tree.yaml` file with your desired decision nodes and choices. To add or remove AI tools, update the `tools` vector in the `run_decision` function.
