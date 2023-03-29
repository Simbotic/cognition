// YAML prompt_decision template object
pub struct DecisionPromptTemplate(String);

impl DecisionPromptTemplate {
    pub fn new(content: String) -> Self {
        Self(content)
    }

    // Format the decision prompt template with the given parameters
    pub fn format(
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
