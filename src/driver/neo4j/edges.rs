//! Edge CRUD operations for the Neo4j driver.

use chrono::{DateTime, Utc};
use neo4rs::{query, Graph};
use uuid::Uuid;

use crate::edges::{CommunityEdge, EntityEdge, EpisodicEdge};
use crate::errors::Result;

use super::convert::{
    driver_err, extract_opt_embedding, opt_dt_param, opt_embedding_param, opt_str_param,
    parse_dt, parse_opt_dt, parse_uuid,
};

// ── EntityEdge ────────────────────────────────────────────────────────────────

/// Build an `EntityEdge` from a row that returns individual fields.
///
/// Expected column aliases (all prefixed with the property name):
/// `uuid`, `source_node_uuid`, `target_node_uuid`, `name`, `fact`,
/// `fact_embedding`, `episodes_json`, `valid_at`, `invalid_at`,
/// `created_at`, `expired_at`, `weight`, `attributes`, `group_id`.
pub(super) fn row_to_entity_edge(row: &neo4rs::Row) -> Result<EntityEdge> {
    let uuid = parse_uuid(&row.get::<String>("uuid").map_err(driver_err)?)?;
    let source_node_uuid =
        parse_uuid(&row.get::<String>("source_node_uuid").map_err(driver_err)?)?;
    let target_node_uuid =
        parse_uuid(&row.get::<String>("target_node_uuid").map_err(driver_err)?)?;
    let name = row.get::<String>("name").map_err(driver_err)?;
    let fact = row.get::<String>("fact").map_err(driver_err)?;
    let fact_embedding =
        extract_opt_embedding(row.get::<Vec<f64>>("fact_embedding").map_err(driver_err));
    let episodes_json = row.get::<String>("episodes_json").map_err(driver_err)?;
    let episode_strs: Vec<String> = serde_json::from_str(&episodes_json)?;
    let episodes: Vec<Uuid> = episode_strs
        .iter()
        .map(|s| parse_uuid(s))
        .collect::<Result<_>>()?;
    let valid_at = parse_opt_dt(row.get::<String>("valid_at").ok())?;
    let invalid_at = parse_opt_dt(row.get::<String>("invalid_at").ok())?;
    let created_at = parse_dt(&row.get::<String>("created_at").map_err(driver_err)?)?;
    let expired_at = parse_opt_dt(row.get::<String>("expired_at").ok())?;
    let weight = row.get::<f64>("weight").map_err(driver_err)?;
    let attributes_json = row.get::<String>("attributes").map_err(driver_err)?;
    let attributes: serde_json::Value = serde_json::from_str(&attributes_json)?;
    let group_id = row.get::<String>("group_id").ok();

    Ok(EntityEdge {
        uuid,
        source_node_uuid,
        target_node_uuid,
        name,
        fact,
        fact_embedding,
        episodes,
        valid_at,
        invalid_at,
        created_at,
        expired_at,
        weight,
        attributes,
        group_id,
    })
}

/// Cypher fragment that returns all EntityEdge fields from relationship alias `r`.
const ENTITY_EDGE_RETURN: &str = "RETURN r.uuid AS uuid, \
       r.source_node_uuid AS source_node_uuid, \
       r.target_node_uuid AS target_node_uuid, \
       r.name AS name, \
       r.fact AS fact, \
       r.fact_embedding AS fact_embedding, \
       r.episodes_json AS episodes_json, \
       r.valid_at AS valid_at, \
       r.invalid_at AS invalid_at, \
       r.created_at AS created_at, \
       r.expired_at AS expired_at, \
       r.weight AS weight, \
       r.attributes AS attributes, \
       r.group_id AS group_id";

pub(super) async fn save_entity_edge(graph: &Graph, edge: &EntityEdge) -> Result<()> {
    let episodes_json = serde_json::to_string(
        &edge
            .episodes
            .iter()
            .map(|u| u.to_string())
            .collect::<Vec<_>>(),
    )?;
    let attributes_json = edge.attributes.to_string();

    let q = query(
        "MATCH (a {uuid: $source_node_uuid}), (b {uuid: $target_node_uuid}) \
         MERGE (a)-[r:RELATES_TO {uuid: $uuid}]->(b) \
         SET r.source_node_uuid = $source_node_uuid, \
             r.target_node_uuid = $target_node_uuid, \
             r.name = $name, \
             r.fact = $fact, \
             r.fact_embedding = $fact_embedding, \
             r.episodes_json = $episodes_json, \
             r.valid_at = $valid_at, \
             r.invalid_at = $invalid_at, \
             r.created_at = $created_at, \
             r.expired_at = $expired_at, \
             r.weight = $weight, \
             r.attributes = $attributes, \
             r.group_id = $group_id",
    )
    .param("uuid", edge.uuid.to_string())
    .param("source_node_uuid", edge.source_node_uuid.to_string())
    .param("target_node_uuid", edge.target_node_uuid.to_string())
    .param("name", edge.name.clone())
    .param("fact", edge.fact.clone())
    .param("fact_embedding", opt_embedding_param(&edge.fact_embedding))
    .param("episodes_json", episodes_json)
    .param("valid_at", opt_dt_param(edge.valid_at))
    .param("invalid_at", opt_dt_param(edge.invalid_at))
    .param("created_at", edge.created_at.to_rfc3339())
    .param("expired_at", opt_dt_param(edge.expired_at))
    .param("weight", edge.weight)
    .param("attributes", attributes_json)
    .param("group_id", opt_str_param(&edge.group_id));

    graph.run(q).await.map_err(driver_err)
}

pub(super) async fn get_entity_edge(graph: &Graph, uuid: &Uuid) -> Result<Option<EntityEdge>> {
    let cypher = format!(
        "MATCH ()-[r:RELATES_TO {{uuid: $uuid}}]->() {}",
        ENTITY_EDGE_RETURN
    );
    let mut stream = graph
        .execute(query(&cypher).param("uuid", uuid.to_string()))
        .await
        .map_err(driver_err)?;

    if let Some(row) = stream.next().await.map_err(driver_err)? {
        Ok(Some(row_to_entity_edge(&row)?))
    } else {
        Ok(None)
    }
}

pub(super) async fn get_entity_edges_between(
    graph: &Graph,
    source: &Uuid,
    target: &Uuid,
) -> Result<Vec<EntityEdge>> {
    let cypher = format!(
        "MATCH ({{uuid: $source_uuid}})-[r:RELATES_TO]-({{uuid: $target_uuid}}) {}",
        ENTITY_EDGE_RETURN
    );
    let mut stream = graph
        .execute(
            query(&cypher)
                .param("source_uuid", source.to_string())
                .param("target_uuid", target.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut edges = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        edges.push(row_to_entity_edge(&row)?);
    }
    Ok(edges)
}

pub(super) async fn invalidate_edge(
    graph: &Graph,
    uuid: &Uuid,
    invalid_at: DateTime<Utc>,
) -> Result<()> {
    graph
        .run(
            query(
                "MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() \
                 SET r.invalid_at = $invalid_at",
            )
            .param("uuid", uuid.to_string())
            .param("invalid_at", invalid_at.to_rfc3339()),
        )
        .await
        .map_err(driver_err)
}

/// Return all active (non-invalidated) entity edges belonging to `group_id`.
pub(super) async fn list_entity_edges(
    graph: &Graph,
    group_id: &str,
) -> Result<Vec<EntityEdge>> {
    let cypher = format!(
        "MATCH ()-[r:RELATES_TO]->() \
         WHERE r.group_id = $group_id AND r.invalid_at IS NULL \
         {}",
        ENTITY_EDGE_RETURN
    );
    let mut stream = graph
        .execute(query(&cypher).param("group_id", group_id.to_string()))
        .await
        .map_err(driver_err)?;

    let mut edges = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        edges.push(row_to_entity_edge(&row)?);
    }
    Ok(edges)
}

// ── EpisodicEdge ──────────────────────────────────────────────────────────────

pub(super) async fn save_episodic_edge(graph: &Graph, edge: &EpisodicEdge) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (a {uuid: $source_node_uuid}), (b {uuid: $target_node_uuid}) \
                 MERGE (a)-[r:MENTIONS {uuid: $uuid}]->(b) \
                 SET r.source_node_uuid = $source_node_uuid, \
                     r.target_node_uuid = $target_node_uuid, \
                     r.created_at = $created_at",
            )
            .param("uuid", edge.uuid.to_string())
            .param("source_node_uuid", edge.source_node_uuid.to_string())
            .param("target_node_uuid", edge.target_node_uuid.to_string())
            .param("created_at", edge.created_at.to_rfc3339()),
        )
        .await
        .map_err(driver_err)
}

// ── CommunityEdge ─────────────────────────────────────────────────────────────

pub(super) async fn save_community_edge(graph: &Graph, edge: &CommunityEdge) -> Result<()> {
    graph
        .run(
            query(
                "MATCH (a {uuid: $source_node_uuid}), (b {uuid: $target_node_uuid}) \
                 MERGE (a)-[r:HAS_MEMBER {uuid: $uuid}]->(b) \
                 SET r.source_node_uuid = $source_node_uuid, \
                     r.target_node_uuid = $target_node_uuid, \
                     r.created_at = $created_at",
            )
            .param("uuid", edge.uuid.to_string())
            .param("source_node_uuid", edge.source_node_uuid.to_string())
            .param("target_node_uuid", edge.target_node_uuid.to_string())
            .param("created_at", edge.created_at.to_rfc3339()),
        )
        .await
        .map_err(driver_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edges::EntityEdge;

    fn make_edge() -> EntityEdge {
        EntityEdge {
            uuid: Uuid::new_v4(),
            source_node_uuid: Uuid::new_v4(),
            target_node_uuid: Uuid::new_v4(),
            name: "KNOWS".into(),
            fact: "Alice knows Bob".into(),
            fact_embedding: None,
            episodes: vec![],
            valid_at: None,
            invalid_at: None,
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
            expired_at: None,
            weight: 1.0,
            attributes: serde_json::Value::Null,
            group_id: None,
        }
    }

    #[test]
    fn episodes_json_roundtrip() {
        let ep1 = Uuid::new_v4();
        let ep2 = Uuid::new_v4();
        let episodes = vec![ep1, ep2];
        let json = serde_json::to_string(
            &episodes
                .iter()
                .map(|u| u.to_string())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let strs: Vec<String> = serde_json::from_str(&json).unwrap();
        let recovered: Vec<Uuid> = strs.iter().map(|s| parse_uuid(s).unwrap()).collect();
        assert_eq!(recovered, episodes);
    }

    #[test]
    fn attributes_json_roundtrip() {
        let edge = make_edge();
        let json = edge.attributes.to_string();
        let recovered: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered, edge.attributes);
    }

    #[test]
    fn opt_embedding_none_produces_null() {
        let p = opt_embedding_param(&None);
        assert!(matches!(p, neo4rs::BoltType::Null(_)));
    }

    #[test]
    fn opt_embedding_some_produces_list() {
        let v = vec![0.1_f32, 0.2, 0.3];
        let p = opt_embedding_param(&Some(v));
        assert!(matches!(p, neo4rs::BoltType::List(_)));
    }
}
