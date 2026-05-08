use anyhow::Result;
use secall_core::{
    store::{get_default_db_path, Database},
    vault::Config,
};

pub async fn run(date: Option<String>) -> Result<()> {
    let config = Config::load_or_default();
    let db = Database::open(&get_default_db_path())?;

    // 날짜 결정 (기본: 오늘)
    let target_date = match date {
        Some(d) => d,
        None => {
            let tz = config.timezone();
            chrono::Utc::now()
                .with_timezone(&tz)
                .format("%Y-%m-%d")
                .to_string()
        }
    };

    let sessions = db.get_sessions_for_date(&target_date)?;
    if sessions.is_empty() {
        eprintln!("No sessions found for {}", target_date);
        return Ok(());
    }

    // 자동화/노이즈 세션 필터링, 최소 2턴 이상
    let meaningful: Vec<_> = sessions
        .iter()
        .filter(|(_, _, _, turns, _, stype)| *turns >= 2 && stype != "automated")
        .collect();

    if meaningful.is_empty() {
        eprintln!(
            "No meaningful sessions for {} (all automated or < 2 turns)",
            target_date
        );
        return Ok(());
    }

    // 프로젝트별 그룹핑
    let mut by_project: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

    let session_ids: Vec<String> = meaningful
        .iter()
        .map(|(id, _, _, _, _, _)| id.clone())
        .collect();

    for (_id, project, summary, turns, tools, _) in &meaningful {
        let proj = project.as_deref().unwrap_or("(기타)").to_string();
        let summary_text = summary
            .as_deref()
            .unwrap_or("")
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(150)
            .collect::<String>();

        // 요약이 노이즈인 경우 스킵
        if summary_text.starts_with("Analyze the following")
            || summary_text.starts_with("<environment_context>")
            || summary_text.starts_with("<local-command-caveat>")
        {
            continue;
        }

        let tools_str = tools.as_deref().unwrap_or("[]");
        let entry = format!("- ({turns}턴, 도구:{tools_str}) {summary_text}");
        by_project.entry(proj).or_default().push(entry);
    }

    if by_project.is_empty() {
        eprintln!("No usable session summaries for {}", target_date);
        return Ok(());
    }

    // 시맨틱 토픽 조회 (graph에서)
    let topics = db.get_topics_for_sessions(&session_ids)?;
    let topic_labels: Vec<String> = topics
        .iter()
        .filter_map(|(_, t)| t.strip_prefix("topic:").map(|s| s.to_string()))
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    // 프롬프트 구성
    let mut project_sections = String::new();
    for (proj, entries) in &by_project {
        project_sections.push_str(&format!("### {proj}\n"));
        for e in entries {
            project_sections.push_str(e);
            project_sections.push('\n');
        }
        project_sections.push('\n');
    }

    let topics_line = if topic_labels.is_empty() {
        String::new()
    } else {
        format!("주요 토픽: {}\n\n", topic_labels.join(", "))
    };

    let total = meaningful.len();
    let automated = sessions.len() - meaningful.len();

    let user_prompt = format!(
        "날짜: {target_date}\n총 세션: {total}개 (자동화 제외: {automated}개)\n{topics_line}\
         프로젝트별 작업 내역:\n{project_sections}\n\
         위 내용을 바탕으로 자연스러운 한국어 개발 작업 일지를 작성해주세요.\n\
         형식: 마크다운, 프로젝트별 섹션, 간결하게 (200자 이내)"
    );

    let system_prompt = "당신은 개발자의 작업 일지를 작성하는 도우미입니다. \
        주어진 세션 요약을 바탕으로 그날 무엇을 했는지 자연스러운 한국어로 정리해주세요. \
        과장하지 말고 실제 작업 내용을 간결하게 서술하세요.";

    // LLM 백엔드로 일기 생성
    let body = if config.graph.semantic_backend == "ollama" {
        let base_url = config
            .graph
            .ollama_url
            .as_deref()
            .unwrap_or("http://localhost:11434");
        let model = config.graph.ollama_model.as_deref().unwrap_or("gemma4:e4b");
        eprintln!("Generating work log with {} ({})...", model, target_date);
        match call_ollama(base_url, model, system_prompt, &user_prompt).await {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Ollama failed ({}), using template output", e);
                generate_template(&target_date, &by_project, &topic_labels, total)
            }
        }
    } else if config.graph.semantic_backend == "gemini" {
        let api_key = config
            .graph
            .gemini_api_key
            .clone()
            .or_else(|| std::env::var("SECALL_GEMINI_API_KEY").ok())
            .ok_or_else(|| anyhow::anyhow!("gemini api key not set"))?;
        let model = config
            .graph
            .gemini_model
            .as_deref()
            .unwrap_or("gemini-2.5-flash")
            .to_string();
        eprintln!("Generating work log with {} ({})...", model, target_date);
        let full_prompt = format!("{}\n\n{}", system_prompt, user_prompt);
        match call_gemini(&full_prompt, &api_key, &model).await {
            Ok(text) => text,
            Err(e) => {
                eprintln!("Gemini failed ({}), using template output", e);
                generate_template(&target_date, &by_project, &topic_labels, total)
            }
        }
    } else {
        generate_template(&target_date, &by_project, &topic_labels, total)
    };

    // 결과 출력
    println!("{}", body);

    // vault에 저장 — host suffix 로 머신간 conflict 회피
    let log_dir = config.vault.path.join("log");
    std::fs::create_dir_all(&log_dir)?;
    let host = gethostname::gethostname()
        .to_string_lossy()
        .split('.')
        .next()
        .unwrap_or("unknown")
        .replace(['/', '\\', ':', '*', '?', '"', '<', '>', '|'], "_");
    let log_path = log_dir.join(format!("{}--{}.md", target_date, host));
    std::fs::write(&log_path, &body)?;
    eprintln!("Saved to {}", log_path.display());

    Ok(())
}

async fn call_ollama(base_url: &str, model: &str, system: &str, user: &str) -> Result<String> {
    let url = format!("{}/api/chat", base_url.trim_end_matches('/'));
    let payload = serde_json::json!({
        "model": model,
        "stream": false,
        "options": {"temperature": 0.3},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let resp = client.post(&url).json(&payload).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("Ollama error {}", resp.status());
    }

    #[derive(serde::Deserialize)]
    struct OllamaResp {
        message: OllamaMsg,
    }
    #[derive(serde::Deserialize)]
    struct OllamaMsg {
        content: String,
    }

    let r: OllamaResp = resp.json().await?;
    Ok(r.message.content)
}

async fn call_gemini(prompt: &str, api_key: &str, model: &str) -> Result<String> {
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let payload = serde_json::json!({
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1024
        }
    });

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("gemini api error {}: {}", status, text);
    }

    let data: serde_json::Value = resp.json().await?;
    let text = data["candidates"][0]["content"]["parts"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(text)
}

pub(crate) fn generate_template(
    date: &str,
    by_project: &std::collections::BTreeMap<String, Vec<String>>,
    topics: &[String],
    total: usize,
) -> String {
    let mut out = format!("# {date} 작업 일지\n\n");
    for (proj, entries) in by_project {
        out.push_str(&format!("## {proj}\n"));
        for e in entries {
            out.push_str(e);
            out.push('\n');
        }
        out.push('\n');
    }
    if !topics.is_empty() {
        out.push_str(&format!("**주요 토픽**: {}\n\n", topics.join(", ")));
    }
    out.push_str(&format!("*총 {total}개 세션*\n"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn test_generate_template_basic() {
        let mut by_project = BTreeMap::new();
        by_project.insert(
            "seCall".to_string(),
            vec!["- (5턴, 도구:[Edit]) Add feature X".to_string()],
        );
        let topics = vec!["rust".to_string(), "async".to_string()];
        let result = generate_template("2026-04-13", &by_project, &topics, 3);

        assert!(result.contains("# 2026-04-13 작업 일지"));
        assert!(result.contains("## seCall"));
        assert!(result.contains("Add feature X"));
        assert!(result.contains("**주요 토픽**: rust, async"));
        assert!(result.contains("*총 3개 세션*"));
    }

    #[test]
    fn test_generate_template_no_topics() {
        let mut by_project = BTreeMap::new();
        by_project.insert(
            "other".to_string(),
            vec!["- (2턴, 도구:[]) Fix bug".to_string()],
        );
        let result = generate_template("2026-04-12", &by_project, &[], 1);

        assert!(result.contains("# 2026-04-12 작업 일지"));
        assert!(!result.contains("주요 토픽"));
        assert!(result.contains("*총 1개 세션*"));
    }

    #[test]
    fn test_generate_template_multiple_projects() {
        let mut by_project = BTreeMap::new();
        by_project.insert("A".to_string(), vec!["- entry A".to_string()]);
        by_project.insert(
            "B".to_string(),
            vec!["- entry B1".to_string(), "- entry B2".to_string()],
        );
        let result = generate_template("2026-01-01", &by_project, &[], 5);

        // BTreeMap이므로 A가 B보다 먼저 나와야 함
        let a_pos = result.find("## A").unwrap();
        let b_pos = result.find("## B").unwrap();
        assert!(a_pos < b_pos);
        assert!(result.contains("entry B2"));
    }
}
