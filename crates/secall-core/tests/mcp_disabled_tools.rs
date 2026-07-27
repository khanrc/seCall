//! `[mcp] disabled_tools` — removing a tool from the router.
//!
//! rmcp's `ToolRouter::list_all` (which builds `tools/list`), `has_route` and
//! `call` all read the same `self.map`, so `remove_route` takes a tool out of
//! advertisement and dispatch together. These tests assert on the advertised
//! set, which is that shared map's contents.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use secall_core::{
    mcp::SeCallMcpServer,
    search::{Bm25Indexer, LinderaKoTokenizer, SearchEngine},
    store::Database,
};

const ALL_TOOLS: [&str; 5] = ["get", "graph_query", "recall", "status", "wiki_search"];

fn make_server() -> SeCallMcpServer {
    let db = Database::open_memory().expect("open in-memory db");
    let tok = LinderaKoTokenizer::new().expect("tokenizer init");
    let engine = SearchEngine::new(Bm25Indexer::new(Box::new(tok)), None);
    SeCallMcpServer::new(
        Arc::new(Mutex::new(db)),
        Arc::new(engine),
        PathBuf::from("/nonexistent-vault"),
    )
}

#[test]
fn advertises_every_tool_by_default() {
    let names = make_server().tool_names();
    assert_eq!(names, ALL_TOOLS.map(str::to_string).to_vec());
}

#[test]
fn empty_denylist_is_a_no_op() {
    let names = make_server()
        .with_disabled_tools(&[])
        .expect("empty denylist")
        .tool_names();
    assert_eq!(names, ALL_TOOLS.map(str::to_string).to_vec());
}

#[test]
fn disabled_tools_drop_out_and_siblings_survive() {
    let disabled: Vec<String> = ["wiki_search", "graph_query", "status"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let names = make_server()
        .with_disabled_tools(&disabled)
        .expect("known tool names")
        .tool_names();

    assert_eq!(names, vec!["get".to_string(), "recall".to_string()]);
}

#[test]
fn unknown_tool_name_is_rejected() {
    let err = make_server()
        .with_disabled_tools(&["recal".to_string()])
        .err()
        .expect("typo must not be silently ignored");

    let msg = err.to_string();
    assert!(msg.contains("recal"), "error should name the bad key: {msg}");
    assert!(
        msg.contains("recall"),
        "error should list the known tools: {msg}"
    );
}

#[test]
fn unknown_name_rejected_even_alongside_valid_ones() {
    assert!(
        make_server()
            .with_disabled_tools(&["status".to_string(), "nope".to_string()])
            .is_err(),
        "one bad name fails the whole list"
    );
}
