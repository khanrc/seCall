use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum QueryType {
    /// BM25 + vector via reciprocal rank fusion (default — best for natural language paraphrase queries).
    #[default]
    Hybrid,
    /// BM25 exact match only — use for strong-IDF identifier queries (function names, ticket IDs, file paths).
    Keyword,
    /// Vector similarity only — use when the query is a paraphrase with no expected exact-token match.
    Semantic,
    /// Date filter: today, yesterday, last week, since YYYY-MM-DD. Does not dispatch a search by itself; pair with another type.
    Temporal,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct QueryItem {
    /// Search mode. Omit (or use "hybrid") for default RRF fusion. Use "keyword" / "semantic" only when you specifically want a single-modal lookup. "temporal" sets a date filter.
    #[serde(rename = "type", default)]
    pub query_type: QueryType,
    /// The search query string
    pub query: String,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecallParams {
    /// Search queries array. Each item defaults to "hybrid" mode (BM25 + vector merged via RRF). Mix types only when you need a specific backend or a temporal filter alongside.
    pub queries: Vec<QueryItem>,
    /// Filter by project name
    pub project: Option<String>,
    /// Filter by agent: claude-code, codex, gemini-cli
    pub agent: Option<String>,
    /// Max results (default 10)
    pub limit: Option<usize>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GetParams {
    /// Session ID or session_id:turn_index
    pub id: String,
    /// Return full markdown content (default: metadata + summary)
    pub full: Option<bool>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct StatusParams {}

#[derive(Debug, Deserialize, Serialize, JsonSchema, Default, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum WikiSearchMode {
    #[default]
    Keyword,
    Semantic,
    Hybrid,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct WikiSearchParams {
    /// Search query matched against wiki filename and content
    pub query: String,
    /// Filter by wiki category: projects, topics, decisions (optional)
    pub category: Option<String>,
    /// Max results (default 5)
    pub limit: Option<usize>,
    /// Search mode: keyword(default), semantic, hybrid
    #[serde(default)]
    pub mode: Option<WikiSearchMode>,
}

#[derive(Debug, Deserialize, JsonSchema)]
pub struct GraphQueryParams {
    /// Node ID to query (e.g., "project:tunaflow", "tool:Edit", "session:abc12345")
    pub node_id: String,
    /// Max traversal depth (default: 1)
    pub depth: Option<usize>,
    /// Filter by relation type (e.g., "belongs_to", "uses_tool", "same_project")
    pub relation: Option<String>,
}
