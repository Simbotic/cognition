# Cognition
## Cognitive decision-making system

Cognition is a decision-making tool that utilizes Large Language Models (LLMs) to interact with users and help them make decisions based on a predefined decision tree. The system also integrates with other AI tools like Wolfram|Alpha to answer factual and mathematical questions.

## Features

- Predefined decision tree in YAML format
- Integration with Large Language Models (LLMs) for decision-making
- Recursive prediction of user responses
- Integration with Wolfram|Alpha for answering factual and mathematical questions
- Supports:
    - OpenAI davinci-003 LLM
    - LLaMA models, including 4-bit modes

## Installation

To install and run the Cognition System, you'll need Rust installed on your machine. If you don't have Rust installed, you can follow the instructions at https://www.rust-lang.org/tools/install.

### Clone the repository

```
git clone https://github.com/Simbotic/cognition.git
cd cognition
```

### Configure API keys

You'll need to set environment variables for the OpenAI and Wolfram|Alpha API keys.
```
export OPENAI_API_KEY="your_openai_api_key"
export WOLFRAM_APP_ID="your_wolfram_app_id"
export TEXTGEN_SERVER="your_textgen_server"
```

### Build and run

```
cargo run --release -p cognition-cli
```

## Usage

Once you've built and run Cognition, you'll be prompted with a series of questions and choices. You can navigate the decision tree by typing your choice and pressing Enter. To exit the system, type "exit" when prompted.

## Customization

To customize the decision tree, modify the `decision_tree.yaml` file with your desired decision nodes and choices. To add or remove AI tools, update the `tools` vector in the `run_decision` function.
