use super::*;

pub struct Signal {
    pub id: String,
    pub name: String,
    pub description: String,
    pub signal: String,
}

#[async_trait(?Send)]
impl Tool for Signal {
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
        debug!("{}: {}", self.id, input);
        Ok(Some(ToolResponse {
            id: self.id.clone(),
            response: self.signal.clone(),
        }))
    }
}
