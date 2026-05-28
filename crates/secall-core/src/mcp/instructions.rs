use crate::store::db::Database;

pub fn build_instructions(db: &Database) -> String {
    let session_count = db.count_sessions().unwrap_or(0);
    let projects = db.list_projects().unwrap_or_default();
    let agents = db.list_agents().unwrap_or_default();
    let has_embeddings = db.has_embeddings().unwrap_or(false);
    let graph_stats = db.graph_stats().unwrap_or_default();

    let graph_section = if graph_stats.node_count > 0 {
        format!(
            r#"
## Graph
- Use `graph_query` to explore relationships between sessions, projects, agents, and tools.
- Node ID format: "project:{{name}}", "agent:{{kind}}", "tool:{{name}}", "session:{{id}}"
- Example: graph_query(node_id="project:tunaflow", depth=2)
- Graph contains {} nodes and {} edges.
"#,
            graph_stats.node_count, graph_stats.edge_count
        )
    } else {
        String::new()
    };

    let graph_tool_line = if graph_stats.node_count > 0 {
        "\n- `graph_query` — explore knowledge graph relationships (session/project/agent/tool nodes)"
    } else {
        ""
    };

    format!(
        r#"seCall — Agent Session Search Engine

Index contains {session_count} sessions across {project_count} projects.
Projects: {projects}
Agents: {agents}
Vector search: {vector_status}

## Usage Tips
- `recall` defaults to hybrid mode (BM25 + vector, merged via RRF). Omit `type` for the default — it's the right choice for most paraphrase / conceptual queries.
- Use `type: "keyword"` only when the query is a strong-IDF identifier (function name, ticket ID, file path) where vector search adds no signal.
- Use `type: "semantic"` only when the query is pure paraphrase and you want to skip BM25 ranking.
- Pair `type: "temporal"` (e.g. "yesterday", "since 2026-05-01") with another item to filter by date window.
- Use `get` with session_id:N to read a specific turn.
- Filter by project or agent when searching across many sessions.

## Tools
- `recall` — search session turns (hybrid by default; keyword / semantic / temporal as overrides)
- `get` — retrieve a specific session or turn by ID
- `status` — show index health
- `wiki_search` — search wiki knowledge pages by query; optional `category` filter (projects/topics/decisions){graph_tool_line}

## Example Queries
- Hybrid (default — recommended): {{"queries": [{{"query": "Korean tokenizer comparison"}}]}}
- Keyword only (identifier): {{"queries": [{{"type": "keyword", "query": "SQLite FTS5"}}]}}
- Semantic only (paraphrase): {{"queries": [{{"type": "semantic", "query": "how to design database schema"}}]}}
- Multiple hybrid queries: {{"queries": [{{"query": "RRF fusion ranking"}}, {{"query": "vector similarity merge"}}]}}
- Temporal filter: {{"queries": [{{"type": "temporal", "query": "yesterday"}}, {{"query": "bugfix"}}]}}
- Wiki: {{"query": "tunadish", "category": "projects", "limit": 3}}
{graph_section}"#,
        session_count = session_count,
        project_count = projects.len(),
        projects = if projects.is_empty() {
            "(none)".to_string()
        } else {
            projects.join(", ")
        },
        agents = if agents.is_empty() {
            "(none)".to_string()
        } else {
            agents.join(", ")
        },
        vector_status = if has_embeddings {
            "enabled"
        } else {
            "disabled (run `secall embed`)"
        },
        graph_tool_line = graph_tool_line,
        graph_section = graph_section,
    )
}
