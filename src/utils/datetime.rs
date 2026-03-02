//! DateTime parsing and formatting helpers for bi-temporal Neo4j timestamps.

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeZone, Utc};

/// Parse a datetime string in various common formats into a UTC [`DateTime`].
///
/// Supported formats (attempted in order):
/// 1. RFC 3339 / ISO 8601 with timezone: `"2024-01-15T10:30:00Z"`, `"2024-01-15T10:30:00+05:00"`
/// 2. Neo4j nanosecond format: `"2024-01-15T10:30:00.000000000Z"`
/// 3. ISO 8601 without timezone (assumed UTC): `"2024-01-15T10:30:00"`
/// 4. ISO 8601 with sub-seconds but no timezone: `"2024-01-15T10:30:00.123"`
/// 5. Date only (midnight UTC): `"2024-01-15"`
/// 6. US date format (midnight UTC): `"01/15/2024"`
///
/// Returns `None` for empty input or unrecognised formats.
pub fn parse_flexible_datetime(s: &str) -> Option<DateTime<Utc>> {
    if s.is_empty() {
        return None;
    }

    // 1. RFC 3339 (covers nanosecond Neo4j format with Z suffix too).
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }

    // 2. ISO 8601 with sub-seconds but no timezone.
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(Utc.from_utc_datetime(&ndt));
    }

    // 3. ISO 8601 without sub-seconds, no timezone.
    if let Ok(ndt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S") {
        return Some(Utc.from_utc_datetime(&ndt));
    }

    // 4. Date only (midnight UTC).
    if let Ok(nd) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return nd
            .and_hms_opt(0, 0, 0)
            .map(|ndt| Utc.from_utc_datetime(&ndt));
    }

    // 5. US date format MM/DD/YYYY (midnight UTC).
    if let Ok(nd) = NaiveDate::parse_from_str(s, "%m/%d/%Y") {
        return nd
            .and_hms_opt(0, 0, 0)
            .map(|ndt| Utc.from_utc_datetime(&ndt));
    }

    None
}

/// Format a [`DateTime<Utc>`] as a Neo4j Cypher datetime literal.
///
/// Output format: `"2024-01-15T10:30:00.000000000Z"` (ISO 8601, nanosecond precision, UTC).
pub fn format_neo4j_datetime(dt: &DateTime<Utc>) -> String {
    dt.format("%Y-%m-%dT%H:%M:%S%.9fZ").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_parse_rfc3339_utc() {
        let dt = parse_flexible_datetime("2024-01-15T10:30:00Z").expect("should parse");
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 15);
        assert_eq!(dt.hour(), 10);
        assert_eq!(dt.minute(), 30);
        assert_eq!(dt.second(), 0);
    }

    #[test]
    fn test_parse_rfc3339_with_offset() {
        // +05:00 offset → 10:30 local = 05:30 UTC
        let dt = parse_flexible_datetime("2024-01-15T10:30:00+05:00").expect("should parse");
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 15, 5, 30, 0).unwrap());
    }

    #[test]
    fn test_parse_iso_no_tz() {
        let dt = parse_flexible_datetime("2024-01-15T10:30:00").expect("should parse");
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap());
    }

    #[test]
    fn test_parse_date_only() {
        let dt = parse_flexible_datetime("2024-01-15").expect("should parse");
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap());
    }

    #[test]
    fn test_parse_us_date_format() {
        let dt = parse_flexible_datetime("01/15/2024").expect("should parse");
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 15, 0, 0, 0).unwrap());
    }

    #[test]
    fn test_parse_neo4j_nanosecond() {
        // Neo4j nanosecond format — parsed by rfc3339 since it has Z suffix.
        let dt = parse_flexible_datetime("2024-01-15T10:30:00.000000000Z").expect("should parse");
        assert_eq!(dt, Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap());
    }

    #[test]
    fn test_parse_nanosecond_precision() {
        let dt = parse_flexible_datetime("2024-06-01T12:00:00.123456789Z").expect("should parse");
        assert_eq!(dt.nanosecond(), 123_456_789);
    }

    #[test]
    fn test_parse_invalid() {
        assert!(parse_flexible_datetime("not a date").is_none());
        assert!(parse_flexible_datetime("2024-13-01").is_none());
        assert!(parse_flexible_datetime("hello world").is_none());
    }

    #[test]
    fn test_parse_empty() {
        assert!(parse_flexible_datetime("").is_none());
    }

    #[test]
    fn test_format_neo4j() {
        let dt = Utc.with_ymd_and_hms(2024, 1, 15, 10, 30, 0).unwrap();
        assert_eq!(format_neo4j_datetime(&dt), "2024-01-15T10:30:00.000000000Z");
    }

    #[test]
    fn test_format_neo4j_midnight() {
        let dt = Utc.with_ymd_and_hms(2024, 12, 31, 0, 0, 0).unwrap();
        assert_eq!(format_neo4j_datetime(&dt), "2024-12-31T00:00:00.000000000Z");
    }

    #[test]
    fn test_format_roundtrip() {
        let dt = Utc.with_ymd_and_hms(2024, 6, 15, 8, 45, 30).unwrap();
        let formatted = format_neo4j_datetime(&dt);
        let parsed = parse_flexible_datetime(&formatted).expect("roundtrip should parse");
        assert_eq!(dt, parsed);
    }

    #[test]
    fn test_format_neo4j_nanoseconds_preserved() {
        // Create a DateTime with sub-second precision.
        let base = Utc.with_ymd_and_hms(2024, 3, 1, 12, 0, 0).unwrap();
        let dt = base + chrono::Duration::nanoseconds(500_000_000);
        let formatted = format_neo4j_datetime(&dt);
        assert!(formatted.ends_with('Z'));
        assert!(formatted.contains(".500000000"));
    }
}

// Bring chrono date/time component accessors into scope for tests.
#[cfg(test)]
use chrono::Datelike;
#[cfg(test)]
use chrono::Timelike;
