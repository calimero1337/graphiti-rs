//! Claude LLM client — uses the Claude Agent SDK to call Claude via CLI subprocess.
//!
//! This replaces the OpenAI client for environments without an OpenAI API key.
//! It spawns `claude` CLI processes for each LLM call, using the Agent SDK's
//! `query()` function.
//!
//! Two conceptual "agents" use this client:
//! - **Seven** (entity/edge extraction) — structured JSON output via schema prompts
//! - **Neelix** (community summarization) — free-text generation

use std::sync::atomic::{AtomicU64, Ordering};

use claude_agent_sdk::{ClaudeAgentOptions, PermissionMode};
use tracing::{debug, warn};

use crate::errors::{GraphitiError, Result};
use crate::llm_client::token_tracker::TokenUsage;
use crate::llm_client::{LlmClient, Message, Role};

/// A [`LlmClient`] implementation that uses the Claude CLI via the Agent SDK.
pub struct ClaudeLlmClient {
    /// Model to use (e.g. "claude-sonnet-4-6").
    model: String,
    /// Optional system prompt override.
    system_prompt: Option<String>,
    /// Working directory for the claude process.
    cwd: Option<std::path::PathBuf>,
    /// Approximate token tracking (estimated, not exact).
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
}

impl ClaudeLlmClient {
    /// Create a new Claude LLM client.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            system_prompt: None,
            cwd: None,
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
        }
    }

    /// Set a default system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the working directory for claude processes.
    pub fn with_cwd(mut self, cwd: impl Into<std::path::PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Build options for a query.
    fn build_options(&self, system_override: Option<&str>) -> ClaudeAgentOptions {
        let system = system_override
            .map(|s| s.to_string())
            .or_else(|| self.system_prompt.clone());

        ClaudeAgentOptions {
            model: Some(self.model.clone()),
            system_prompt: system,
            permission_mode: Some(PermissionMode::BypassPermissions),
            max_turns: Some(1),
            cwd: self.cwd.clone(),
            ..Default::default()
        }
    }

    /// Convert a sequence of LLM messages into a single prompt string.
    ///
    /// Returns (system_prompt, user_prompt).
    fn messages_to_prompt(messages: &[Message]) -> (Option<String>, String) {
        let mut system_parts = Vec::new();
        let mut conversation_parts = Vec::new();

        for msg in messages {
            match msg.role {
                Role::System => system_parts.push(msg.content.clone()),
                Role::User => conversation_parts.push(msg.content.clone()),
                Role::Assistant => {
                    // Include assistant context as part of the conversation.
                    conversation_parts
                        .push(format!("[Previous assistant response]: {}", msg.content));
                }
            }
        }

        let system = if system_parts.is_empty() {
            None
        } else {
            Some(system_parts.join("\n\n"))
        };

        let prompt = conversation_parts.join("\n\n");
        (system, prompt)
    }

    /// Track token usage from a result (estimated from string lengths).
    fn track_usage(&self, prompt_len: usize, response_len: usize) {
        // Rough estimate: ~4 chars per token.
        let prompt_tokens = (prompt_len / 4) as u64;
        let completion_tokens = (response_len / 4) as u64;
        self.prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);
    }
}

#[async_trait::async_trait]
impl LlmClient for ClaudeLlmClient {
    async fn generate(&self, messages: &[Message]) -> Result<String> {
        let (system, prompt) = Self::messages_to_prompt(messages);
        let options = self.build_options(system.as_deref());

        debug!(model = %self.model, prompt_len = prompt.len(), "claude generate");

        let result = claude_agent_sdk::query(&prompt, &options)
            .await
            .map_err(|e| GraphitiError::Pipeline(format!("Claude query failed: {e}")))?;

        let response = result.response_text();
        if response.is_empty() {
            warn!("Claude returned empty response");
        }

        self.track_usage(prompt.len(), response.len());
        Ok(response)
    }

    async fn generate_structured_json(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<String> {
        let (system, prompt) = Self::messages_to_prompt(messages);

        // Wrap the system prompt with JSON schema instructions.
        let schema_str = serde_json::to_string_pretty(&schema)
            .unwrap_or_else(|_| schema.to_string());

        let structured_system = format!(
            "{}\n\n\
            CRITICAL: You MUST respond with ONLY valid JSON matching this schema. \
            No markdown, no code fences, no explanation — ONLY the JSON object.\n\n\
            JSON Schema:\n```json\n{schema_str}\n```",
            system.as_deref().unwrap_or("You are a helpful assistant.")
        );

        let options = self.build_options(Some(&structured_system));

        debug!(
            model = %self.model,
            prompt_len = prompt.len(),
            "claude generate_structured_json"
        );

        let result = claude_agent_sdk::query(&prompt, &options)
            .await
            .map_err(|e| GraphitiError::Pipeline(format!("Claude structured query failed: {e}")))?;

        let mut response = result.response_text();

        // Strip markdown code fences if Claude wrapped the JSON.
        response = strip_code_fences(&response);

        if response.is_empty() {
            return Err(GraphitiError::Pipeline(
                "Claude returned empty structured response".into(),
            ));
        }

        // Validate it's actually JSON.
        if let Err(e) = serde_json::from_str::<serde_json::Value>(&response) {
            return Err(GraphitiError::Pipeline(format!(
                "Claude response is not valid JSON: {e}\nResponse: {response}"
            )));
        }

        self.track_usage(prompt.len(), response.len());
        Ok(response)
    }

    fn token_usage(&self) -> TokenUsage {
        let prompt = self.prompt_tokens.load(Ordering::Relaxed);
        let completion = self.completion_tokens.load(Ordering::Relaxed);
        TokenUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
        }
    }

    fn reset_token_usage(&self) {
        self.prompt_tokens.store(0, Ordering::Relaxed);
        self.completion_tokens.store(0, Ordering::Relaxed);
    }
}

/// Strip markdown code fences from a JSON response.
fn strip_code_fences(s: &str) -> String {
    let trimmed = s.trim();
    // Handle ```json ... ``` or ``` ... ```
    if trimmed.starts_with("```") {
        let without_opening = if let Some(rest) = trimmed.strip_prefix("```json") {
            rest
        } else if let Some(rest) = trimmed.strip_prefix("```JSON") {
            rest
        } else {
            trimmed.strip_prefix("```").unwrap_or(trimmed)
        };
        let without_closing = without_opening
            .trim()
            .strip_suffix("```")
            .unwrap_or(without_opening);
        return without_closing.trim().to_string();
    }
    trimmed.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_code_fences_json() {
        let input = "```json\n{\"entities\": []}\n```";
        assert_eq!(strip_code_fences(input), "{\"entities\": []}");
    }

    #[test]
    fn strip_code_fences_plain() {
        let input = "```\n{\"a\": 1}\n```";
        assert_eq!(strip_code_fences(input), "{\"a\": 1}");
    }

    #[test]
    fn strip_code_fences_no_fences() {
        let input = "{\"entities\": []}";
        assert_eq!(strip_code_fences(input), "{\"entities\": []}");
    }

    #[test]
    fn messages_to_prompt_separates_system() {
        let messages = vec![
            Message {
                role: Role::System,
                content: "You extract entities.".into(),
            },
            Message {
                role: Role::User,
                content: "Alice is a software engineer.".into(),
            },
        ];
        let (system, prompt) = ClaudeLlmClient::messages_to_prompt(&messages);
        assert_eq!(system.as_deref(), Some("You extract entities."));
        assert_eq!(prompt, "Alice is a software engineer.");
    }

    #[test]
    fn messages_to_prompt_no_system() {
        let messages = vec![Message {
            role: Role::User,
            content: "Hello".into(),
        }];
        let (system, prompt) = ClaudeLlmClient::messages_to_prompt(&messages);
        assert!(system.is_none());
        assert_eq!(prompt, "Hello");
    }

    #[test]
    fn token_tracking() {
        let client = ClaudeLlmClient::new("test-model");
        client.track_usage(400, 200);
        let usage = client.token_usage();
        assert_eq!(usage.prompt_tokens, 100); // 400/4
        assert_eq!(usage.completion_tokens, 50); // 200/4
        assert_eq!(usage.total_tokens, 150);

        client.reset_token_usage();
        let usage = client.token_usage();
        assert_eq!(usage.total_tokens, 0);
    }

    #[test]
    fn build_options_sets_model_and_permissions() {
        let client = ClaudeLlmClient::new("claude-sonnet-4-6");
        let opts = client.build_options(None);
        assert_eq!(opts.model.as_deref(), Some("claude-sonnet-4-6"));
        assert_eq!(
            opts.permission_mode,
            Some(PermissionMode::BypassPermissions)
        );
        assert_eq!(opts.max_turns, Some(1));
    }
}
