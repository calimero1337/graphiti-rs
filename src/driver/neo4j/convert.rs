//! Type-conversion helpers between Rust types and neo4rs `BoltType`.

use chrono::{DateTime, Utc};
use neo4rs::{BoltNull, BoltType};

use crate::errors::{GraphitiError, Result};

// ── Error wrapping ────────────────────────────────────────────────────────────

pub(super) fn driver_err(e: impl std::fmt::Display) -> GraphitiError {
    GraphitiError::Driver(e.to_string())
}

// ── UUID ──────────────────────────────────────────────────────────────────────

pub(super) fn parse_uuid(s: &str) -> Result<uuid::Uuid> {
    uuid::Uuid::parse_str(s).map_err(driver_err)
}

// ── DateTime ──────────────────────────────────────────────────────────────────

/// Convert `DateTime<Utc>` to an RFC 3339 string parameter.
pub(super) fn dt_param(dt: DateTime<Utc>) -> BoltType {
    BoltType::from(dt.to_rfc3339())
}

/// Convert an optional `DateTime<Utc>` to a string parameter or `BoltNull`.
pub(super) fn opt_dt_param(dt: Option<DateTime<Utc>>) -> BoltType {
    match dt {
        Some(d) => dt_param(d),
        None => BoltType::Null(BoltNull),
    }
}

/// Parse an RFC 3339 string back into `DateTime<Utc>`.
pub(super) fn parse_dt(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|d| d.with_timezone(&Utc))
        .map_err(driver_err)
}

/// Extract an optional `DateTime<Utc>` from a nullable row string field.
///
/// Returns `None` when the field is absent, null, or empty.
pub(super) fn parse_opt_dt(s: Option<String>) -> Result<Option<DateTime<Utc>>> {
    match s {
        Some(ref v) if !v.is_empty() => parse_dt(v).map(Some),
        _ => Ok(None),
    }
}

// ── Optional string ───────────────────────────────────────────────────────────

pub(super) fn opt_str_param(s: &Option<String>) -> BoltType {
    match s {
        Some(v) => BoltType::from(v.clone()),
        None => BoltType::Null(BoltNull),
    }
}

// ── Embedding ─────────────────────────────────────────────────────────────────

/// Convert an optional `Vec<f32>` embedding to a bolt list or `BoltNull`.
pub(super) fn opt_embedding_param(emb: &Option<Vec<f32>>) -> BoltType {
    match emb {
        Some(v) => BoltType::from(v.clone()),
        None => BoltType::Null(BoltNull),
    }
}

/// Extract an optional `Vec<f32>` from a row field that stores a `Vec<f64>`.
///
/// Returns `None` on any deserialization error (field missing / null).
pub(super) fn extract_opt_embedding(raw: Result<Vec<f64>>) -> Option<Vec<f32>> {
    raw.ok().map(|v| v.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn dt_roundtrip() {
        let original = Utc.with_ymd_and_hms(2025, 6, 15, 12, 0, 0).unwrap();
        let param = dt_param(original);
        let s = match param {
            BoltType::String(ref bs) => bs.value.clone(),
            _ => panic!("expected BoltType::String"),
        };
        let recovered = parse_dt(&s).expect("parse_dt failed");
        assert_eq!(recovered, original);
    }

    #[test]
    fn opt_dt_param_some() {
        let dt = Utc::now();
        let p = opt_dt_param(Some(dt));
        assert!(matches!(p, BoltType::String(_)));
    }

    #[test]
    fn opt_dt_param_none() {
        let p = opt_dt_param(None);
        assert!(matches!(p, BoltType::Null(_)));
    }

    #[test]
    fn opt_embedding_param_some() {
        let v = vec![0.1_f32, 0.2, 0.3];
        let p = opt_embedding_param(&Some(v));
        assert!(matches!(p, BoltType::List(_)));
    }

    #[test]
    fn opt_embedding_param_none() {
        let p = opt_embedding_param(&None);
        assert!(matches!(p, BoltType::Null(_)));
    }

    #[test]
    fn parse_uuid_valid() {
        let id = uuid::Uuid::new_v4();
        let recovered = parse_uuid(&id.to_string()).unwrap();
        assert_eq!(recovered, id);
    }

    #[test]
    fn parse_uuid_invalid() {
        assert!(parse_uuid("not-a-uuid").is_err());
    }
}
