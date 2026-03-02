//! LLM client abstraction.
//!
//! Mirrors the Python `graphiti_core.llm_client` module.
//! Provides a trait for calling language models with structured output support.
//!
//! # Implementations
//! - [`openai::OpenAiClient`] — OpenAI GPT-4o (and variants) via `async-openai`.
//!
//! Phase 1 target: OpenAI GPT-4o via `async-openai` with `schemars`-generated JSON schemas.
//! Phase 2: Anthropic Claude, Google Gemini, Groq, Azure OpenAI.

pub mod anthropic;
pub mod claude;
pub mod openai;
pub mod token_tracker;

pub use token_tracker::TokenUsage;

use crate::errors::{GraphitiError, Result};
use serde::de::DeserializeOwned;
use serde::Serialize;

/// A chat message for the LLM conversation.
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

/// Speaker role in a chat conversation.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Trait for LLM clients supporting structured output (JSON schema).
///
/// Implementations must provide [`generate`](LlmClient::generate) (plain text)
/// and [`generate_structured_json`](LlmClient::generate_structured_json)
/// (schema-constrained JSON string).
///
/// The convenience method [`generate_structured`](LlmClient::generate_structured)
/// is provided automatically — it derives the JSON schema from `T` via
/// `schemars`, delegates to `generate_structured_json`, and deserialises the
/// result.  Because it has a generic type parameter it cannot be called through
/// `dyn LlmClient`; callers behind a trait object should use
/// `generate_structured_json` and deserialise manually if needed.
#[async_trait::async_trait]
pub trait LlmClient: Send + Sync {
    /// Send a request and parse the response as plain text.
    async fn generate(&self, messages: &[Message]) -> Result<String>;

    /// Send a request with a JSON-schema constraint and return the raw response string.
    ///
    /// `schema` is a `serde_json::Value` representing the JSON schema that the
    /// model output must conform to.  Implementations should pass this through
    /// to the model's structured-output / JSON-mode mechanism.
    async fn generate_structured_json(
        &self,
        messages: &[Message],
        schema: serde_json::Value,
    ) -> Result<String>;

    /// Send a request and parse the response as a structured JSON type.
    ///
    /// Uses JSON schema derived from `T` (via `schemars`) to constrain the model output.
    ///
    /// This is a provided method that derives the schema, calls
    /// [`generate_structured_json`](LlmClient::generate_structured_json), and
    /// deserialises the result.  Because it is generic over `T` it requires
    /// `Self: Sized` and cannot be called through `dyn LlmClient`.
    async fn generate_structured<T>(&self, messages: &[Message]) -> Result<T>
    where
        T: DeserializeOwned + schemars::JsonSchema + Send,
        Self: Sized,
    {
        let schema = schemars::schema_for!(T);
        let schema_value = serde_json::to_value(&schema)?;
        let raw = self.generate_structured_json(messages, schema_value).await?;
        serde_json::from_str(&raw).map_err(GraphitiError::Serialization)
    }

    /// Return a snapshot of the cumulative token usage for this client.
    ///
    /// The default implementation returns zeroed counters.  Clients that track
    /// real usage (e.g. [`openai::OpenAiClient`]) override this method.
    fn token_usage(&self) -> TokenUsage {
        TokenUsage::default()
    }

    /// Reset the cumulative token counters to zero.
    ///
    /// The default implementation is a no-op.
    fn reset_token_usage(&self) {}
}

/// Call [`LlmClient::generate_structured_json`] through a trait object and
/// deserialise the result into `T`.
///
/// This is the dyn-compatible equivalent of [`LlmClient::generate_structured`],
/// which cannot be called through `dyn LlmClient` because it is generic.
pub async fn generate_structured_via_dyn<T>(
    llm: &dyn LlmClient,
    messages: &[Message],
) -> Result<T>
where
    T: DeserializeOwned + schemars::JsonSchema,
{
    let schema = schemars::schema_for!(T);
    let schema_value = serde_json::to_value(&schema)?;
    let raw = llm.generate_structured_json(messages, schema_value).await?;
    serde_json::from_str(&raw).map_err(GraphitiError::Serialization)
}
