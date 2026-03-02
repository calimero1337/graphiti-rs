//! Text processing utilities.

use regex::Regex;
use std::sync::OnceLock;

static WHITESPACE_RE: OnceLock<Regex> = OnceLock::new();

fn whitespace_re() -> &'static Regex {
    WHITESPACE_RE.get_or_init(|| Regex::new(r"\s+").expect("static regex is valid"))
}

/// Replace consecutive whitespace (spaces, tabs, newlines) with a single space
/// and trim leading/trailing whitespace.
///
/// Returns an empty string for inputs that are entirely whitespace.
pub fn normalize_whitespace(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    whitespace_re().replace_all(s, " ").trim().to_string()
}

/// Truncate `s` to at most `max_len` Unicode scalar values, appending `"..."` if
/// truncation occurred. `max_len` counts characters (not bytes), making this safe
/// for multi-byte UTF-8 content including emoji and CJK characters.
///
/// Special cases:
/// - `max_len == 0` → empty string
/// - `max_len <= 3` → up to `max_len` dots (e.g. `max_len=2` → `".."`)
pub fn truncate_with_ellipsis(s: &str, max_len: usize) -> String {
    if max_len == 0 {
        return String::new();
    }

    let char_count = s.chars().count();
    if char_count <= max_len {
        return s.to_string();
    }

    if max_len <= 3 {
        return ".".repeat(max_len);
    }

    // Find byte offset of the (max_len - 3)th character.
    let keep_chars = max_len - 3;
    let byte_offset = s
        .char_indices()
        .nth(keep_chars)
        .map(|(i, _)| i)
        .unwrap_or(s.len());

    format!("{}...", &s[..byte_offset])
}

/// Extract the first JSON object or array from a potentially markdown-wrapped
/// LLM response.
///
/// Tries, in order:
/// 1. ` ```json ... ``` ` fenced code block
/// 2. ` ``` ... ``` ` fenced code block
/// 3. Bare `{...}` or `[...]` delimited by the first `{`/`[` and last `}`/`]`
///
/// Returns `None` if no JSON-like content is found.
pub fn extract_json_from_response(s: &str) -> Option<&str> {
    // 1. Try ```json fenced block.
    if let Some(inner) = extract_fenced_block(s, "```json") {
        return Some(inner);
    }

    // 2. Try plain ``` fenced block.
    if let Some(inner) = extract_fenced_block(s, "```") {
        return Some(inner);
    }

    // 3. Bare JSON object.
    if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            if end > start {
                return Some(&s[start..=end]);
            }
        }
    }

    // 4. Bare JSON array.
    if let Some(start) = s.find('[') {
        if let Some(end) = s.rfind(']') {
            if end > start {
                return Some(&s[start..=end]);
            }
        }
    }

    None
}

/// Extract content inside a fenced code block starting with `fence`.
fn extract_fenced_block<'a>(s: &'a str, fence: &str) -> Option<&'a str> {
    let start = s.find(fence)?;
    let after_fence = start + fence.len();

    // Skip to end of the opening fence line.
    let newline = s[after_fence..].find('\n')?;
    let content_start = after_fence + newline + 1;

    // Find closing ```.
    let close = s[content_start..].find("```")?;
    let content = s[content_start..content_start + close].trim();

    if content.is_empty() {
        return None;
    }

    Some(content)
}

/// Escape Lucene special characters for safe use in Neo4j full-text index queries.
///
/// Escapes: `+ - ! ( ) { } [ ] ^ " ~ * ? : \ /` and the two-char operators `&&` and `||`.
pub fn lucene_sanitize(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }

    const SPECIAL: &[char] = &[
        '+', '-', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/',
    ];

    let mut result = String::with_capacity(s.len() * 2);
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        // Check two-char sequences first.
        if i + 1 < chars.len() {
            if chars[i] == '&' && chars[i + 1] == '&' {
                result.push_str("\\&&");
                i += 2;
                continue;
            }
            if chars[i] == '|' && chars[i + 1] == '|' {
                result.push_str("\\||");
                i += 2;
                continue;
            }
        }

        let c = chars[i];
        if SPECIAL.contains(&c) {
            result.push('\\');
        }
        result.push(c);
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- normalize_whitespace ---

    #[test]
    fn test_normalize_whitespace_basic() {
        assert_eq!(normalize_whitespace("hello   world"), "hello world");
        assert_eq!(normalize_whitespace("hello\t\tworld"), "hello world");
        assert_eq!(normalize_whitespace("hello\n\nworld"), "hello world");
        assert_eq!(normalize_whitespace("  hello  world  "), "hello world");
    }

    #[test]
    fn test_normalize_whitespace_empty() {
        assert_eq!(normalize_whitespace(""), "");
    }

    #[test]
    fn test_normalize_whitespace_only_whitespace() {
        assert_eq!(normalize_whitespace("   \t\n  "), "");
    }

    #[test]
    fn test_normalize_whitespace_single_word() {
        assert_eq!(normalize_whitespace("hello"), "hello");
    }

    #[test]
    fn test_normalize_whitespace_unicode() {
        // Non-breaking space U+00A0 — \s matches Unicode whitespace in Rust regex.
        assert_eq!(normalize_whitespace("hello\u{00A0}world"), "hello world");
    }

    // --- truncate_with_ellipsis ---

    #[test]
    fn test_truncate_basic() {
        assert_eq!(truncate_with_ellipsis("hello world", 8), "hello...");
    }

    #[test]
    fn test_truncate_short_string() {
        assert_eq!(truncate_with_ellipsis("hi", 10), "hi");
        assert_eq!(truncate_with_ellipsis("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_exact_length() {
        assert_eq!(truncate_with_ellipsis("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_emoji() {
        // Each emoji is one char but multiple bytes.
        // "😀😀😀😀😀" = 5 chars, truncate to 4 → "😀..."
        let s = "😀😀😀😀😀";
        let result = truncate_with_ellipsis(s, 4);
        assert_eq!(result, "😀...");
    }

    #[test]
    fn test_truncate_cjk() {
        // CJK chars are 3 bytes each.
        let s = "你好世界测试";
        let result = truncate_with_ellipsis(s, 5);
        assert_eq!(result, "你好...");
    }

    #[test]
    fn test_truncate_zero_length() {
        assert_eq!(truncate_with_ellipsis("hello", 0), "");
    }

    #[test]
    fn test_truncate_max_len_1() {
        assert_eq!(truncate_with_ellipsis("hello", 1), ".");
    }

    #[test]
    fn test_truncate_max_len_2() {
        assert_eq!(truncate_with_ellipsis("hello", 2), "..");
    }

    #[test]
    fn test_truncate_max_len_3() {
        // "hello" has 5 chars > 3, so truncate to "..."
        assert_eq!(truncate_with_ellipsis("hello", 3), "...");
    }

    #[test]
    fn test_truncate_empty_string() {
        assert_eq!(truncate_with_ellipsis("", 10), "");
    }

    // --- extract_json_from_response ---

    #[test]
    fn test_extract_json_fenced_json() {
        let s = "Here is the result:\n```json\n{\"key\": \"value\"}\n```\nDone.";
        let result = extract_json_from_response(s);
        assert_eq!(result, Some("{\"key\": \"value\"}"));
    }

    #[test]
    fn test_extract_json_fenced_plain() {
        let s = "Result:\n```\n[1, 2, 3]\n```";
        let result = extract_json_from_response(s);
        assert_eq!(result, Some("[1, 2, 3]"));
    }

    #[test]
    fn test_extract_json_bare_object() {
        let s = "The answer is {\"foo\": 42} as shown.";
        let result = extract_json_from_response(s);
        assert_eq!(result, Some("{\"foo\": 42}"));
    }

    #[test]
    fn test_extract_json_bare_array() {
        let s = "Items: [1, 2, 3]";
        let result = extract_json_from_response(s);
        assert_eq!(result, Some("[1, 2, 3]"));
    }

    #[test]
    fn test_extract_json_none() {
        let s = "No JSON here, just plain text.";
        assert_eq!(extract_json_from_response(s), None);
    }

    #[test]
    fn test_extract_json_empty() {
        assert_eq!(extract_json_from_response(""), None);
    }

    #[test]
    fn test_extract_json_empty_fenced_block() {
        let s = "```json\n\n```";
        // Empty fenced block → fall through to bare detection → None.
        assert_eq!(extract_json_from_response(s), None);
    }

    #[test]
    fn test_extract_json_nested_braces() {
        let s = r#"{"outer": {"inner": 1}}"#;
        let result = extract_json_from_response(s);
        assert_eq!(result, Some(r#"{"outer": {"inner": 1}}"#));
    }

    // --- lucene_sanitize ---

    #[test]
    fn test_lucene_sanitize_no_special() {
        assert_eq!(lucene_sanitize("hello world"), "hello world");
    }

    #[test]
    fn test_lucene_sanitize_empty() {
        assert_eq!(lucene_sanitize(""), "");
    }

    #[test]
    fn test_lucene_sanitize_special_chars() {
        assert_eq!(lucene_sanitize("+"), "\\+");
        assert_eq!(lucene_sanitize("-"), "\\-");
        assert_eq!(lucene_sanitize("!"), "\\!");
        assert_eq!(lucene_sanitize("(test)"), "\\(test\\)");
        assert_eq!(lucene_sanitize("a:b"), "a\\:b");
        assert_eq!(lucene_sanitize("a*b"), "a\\*b");
    }

    #[test]
    fn test_lucene_sanitize_double_ampersand() {
        assert_eq!(lucene_sanitize("a&&b"), "a\\&&b");
    }

    #[test]
    fn test_lucene_sanitize_double_pipe() {
        assert_eq!(lucene_sanitize("a||b"), "a\\||b");
    }

    #[test]
    fn test_lucene_sanitize_all_special() {
        let input = r#"+ - ! ( ) { } [ ] ^ " ~ * ? : \ /"#;
        let result = lucene_sanitize(input);
        // Every special char should be escaped.
        assert!(result.contains("\\+"));
        assert!(result.contains("\\-"));
        assert!(result.contains("\\!"));
        assert!(result.contains("\\("));
        assert!(result.contains("\\)"));
    }

    #[test]
    fn test_lucene_sanitize_backslash() {
        assert_eq!(lucene_sanitize("a\\b"), "a\\\\b");
    }

    #[test]
    fn test_lucene_sanitize_slash() {
        assert_eq!(lucene_sanitize("a/b"), "a\\/b");
    }
}
