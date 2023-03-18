use serde_json::Value;
use serde_yaml;

pub fn object_by_path(config: &String, search_path: &str) -> Option<Value> {
    let yaml_value: Value =
        serde_yaml::from_str(&config).expect("Unable to parse the YAML content.");
    let mut current_value = &yaml_value;
    let path_parts: Vec<&str> = search_path.split(".").collect();

    for part in path_parts {
        if let Some(map) = current_value.as_object() {
            if let Some(next_value) = map.get(part) {
                current_value = next_value;
            } else {
                return None;
            }
        } else {
            return None;
        }
    }

    Some(current_value.clone())
}

pub fn string_by_path(config: &String, search_path: &str) -> Option<String> {
    let value = object_by_path(config, search_path);
    if let Some(value) = value {
        if let Some(value) = value.as_str() {
            return Some(value.to_string());
        }
    }
    None
}
