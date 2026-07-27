//! `tools/list` and `tools/call` over the real Streamable-HTTP surface.
//!
//! The unit tests in `mcp_disabled_tools.rs` assert on the server's own
//! router. That is not enough: `#[tool_handler]` defaults to
//! `Self::tool_router()`, a router rebuilt per request, so a handler wired
//! that way answers from a fresh route table and ignores the instance
//! entirely — advertisement and dispatch keep every tool while the instance
//! looks correctly pruned. These tests drive the protocol instead, so that
//! wiring is what gets checked.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use rmcp::transport::streamable_http_server::{
    session::local::LocalSessionManager, StreamableHttpServerConfig, StreamableHttpService,
};
use secall_core::{
    mcp::SeCallMcpServer,
    search::{Bm25Indexer, LinderaKoTokenizer, SearchEngine},
    store::Database,
};
use tower::ServiceExt;

const MCP_PATH: &str = "/mcp";

fn router(disabled: Vec<String>) -> axum::Router {
    let disabled = Arc::new(disabled);
    let service: StreamableHttpService<SeCallMcpServer, LocalSessionManager> =
        StreamableHttpService::new(
            move || -> Result<SeCallMcpServer, std::io::Error> {
                let db = Database::open_memory().map_err(std::io::Error::other)?;
                let tok = LinderaKoTokenizer::new().map_err(std::io::Error::other)?;
                let engine = SearchEngine::new(Bm25Indexer::new(Box::new(tok)), None);
                SeCallMcpServer::new(
                    Arc::new(Mutex::new(db)),
                    Arc::new(engine),
                    PathBuf::from("/nonexistent-vault"),
                )
                .with_disabled_tools(&disabled)
                .map_err(std::io::Error::other)
            },
            Arc::new(LocalSessionManager::default()),
            StreamableHttpServerConfig::default(),
        );
    axum::Router::new().nest_service(MCP_PATH, service)
}

/// One JSON-RPC POST. Returns (status, session id if the server assigned one, body).
async fn post(
    app: &axum::Router,
    session: Option<&str>,
    payload: serde_json::Value,
) -> (StatusCode, Option<String>, String) {
    // Host is required — rmcp rejects a request without one as a bad request
    // (DNS-rebinding guard). A real client always sends it; `Request::builder`
    // does not.
    let mut req = Request::builder()
        .method("POST")
        .uri(MCP_PATH)
        .header("Host", "127.0.0.1")
        .header("Content-Type", "application/json")
        .header("Accept", "application/json, text/event-stream");
    if let Some(sid) = session {
        req = req.header("Mcp-Session-Id", sid);
    }
    let res = app
        .clone()
        .oneshot(req.body(Body::from(payload.to_string())).unwrap())
        .await
        .expect("router oneshot must not fail");

    let status = res.status();
    let sid = res
        .headers()
        .get("Mcp-Session-Id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let bytes = axum::body::to_bytes(res.into_body(), 1 << 20).await.unwrap();
    (status, sid, String::from_utf8_lossy(&bytes).to_string())
}

/// Responses arrive as SSE; pull the last `data:` frame out as JSON.
fn last_frame(body: &str) -> serde_json::Value {
    let line = body
        .lines()
        .filter(|l| l.starts_with("data: {"))
        .next_back()
        .unwrap_or_else(|| panic!("no JSON frame in body:\n{body}"));
    serde_json::from_str(&line["data: ".len()..]).expect("frame parses as JSON")
}

async fn handshake(app: &axum::Router) -> String {
    let (status, sid, body) = post(
        app,
        None,
        serde_json::json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"}
            }
        }),
    )
    .await;
    assert!(status.is_success(), "initialize failed: {status} — {body}");
    let sid = sid.expect("server must assign a session id");

    post(
        app,
        Some(&sid),
        serde_json::json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    )
    .await;
    sid
}

async fn advertised(app: &axum::Router, sid: &str) -> Vec<String> {
    let (_, _, body) = post(
        app,
        Some(sid),
        serde_json::json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
    )
    .await;
    let mut names: Vec<String> = last_frame(&body)["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect();
    names.sort();
    names
}

#[tokio::test]
async fn advertises_every_tool_when_nothing_is_disabled() {
    let app = router(vec![]);
    let sid = handshake(&app).await;
    assert_eq!(
        advertised(&app, &sid).await,
        ["get", "graph_query", "recall", "status", "wiki_search"]
    );
}

#[tokio::test]
async fn disabled_tools_are_not_advertised() {
    let app = router(vec!["wiki_search".into(), "graph_query".into(), "status".into()]);
    let sid = handshake(&app).await;
    assert_eq!(advertised(&app, &sid).await, ["get", "recall"]);
}

#[tokio::test]
async fn calling_a_disabled_tool_is_rejected() {
    let app = router(vec!["wiki_search".into()]);
    let sid = handshake(&app).await;

    let (_, _, body) = post(
        &app,
        Some(&sid),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {"name": "wiki_search", "arguments": {"query": "anything"}}
        }),
    )
    .await;

    let frame = last_frame(&body);
    assert!(
        frame.get("error").is_some(),
        "a removed route must not dispatch, got: {frame}"
    );
}

#[tokio::test]
async fn sibling_tools_still_dispatch() {
    let app = router(vec!["wiki_search".into(), "graph_query".into(), "status".into()]);
    let sid = handshake(&app).await;

    let (_, _, body) = post(
        &app,
        Some(&sid),
        serde_json::json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "recall", "arguments": {"queries": [{"query": "anything"}]}}
        }),
    )
    .await;

    let frame = last_frame(&body);
    assert!(
        frame.get("result").is_some(),
        "recall must still work alongside disabled siblings, got: {frame}"
    );
}
