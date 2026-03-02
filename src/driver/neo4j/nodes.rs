//! Node CRUD operations for the Neo4j driver.

use neo4rs::{query, Graph};
use uuid::Uuid;

use crate::errors::Result;
use crate::nodes::{CommunityNode, EntityNode, EpisodeType, EpisodicNode};

use super::convert::{
    driver_err, extract_opt_embedding, dt_param, opt_embedding_param, parse_dt, parse_uuid,
};

// ── EntityNode ────────────────────────────────────────────────────────────────

/// Extract an `EntityNode` from a single result row.
///
/// Expects columns: `uuid`, `name`, `group_id`, `labels_json`, `summary`,
/// `name_embedding`, `attributes`, `created_at`.
fn row_to_entity_node(row: &neo4rs::Row) -> Result<EntityNode> {
    let uuid = parse_uuid(&row.get::<String>("uuid").map_err(driver_err)?)?;
    let name = row.get::<String>("name").map_err(driver_err)?;
    let group_id = row.get::<String>("group_id").map_err(driver_err)?;
    let labels_json = row.get::<String>("labels_json").map_err(driver_err)?;
    let labels: Vec<String> = serde_json::from_str(&labels_json)?;
    let summary = row.get::<String>("summary").map_err(driver_err)?;
    let name_embedding =
        extract_opt_embedding(row.get::<Vec<f64>>("name_embedding").map_err(driver_err));
    let attributes_json = row.get::<String>("attributes").map_err(driver_err)?;
    let attributes: serde_json::Value = serde_json::from_str(&attributes_json)?;
    let created_at = parse_dt(&row.get::<String>("created_at").map_err(driver_err)?)?;

    Ok(EntityNode {
        uuid,
        name,
        group_id,
        labels,
        summary,
        name_embedding,
        attributes,
        created_at,
    })
}

pub(super) async fn save_entity_node(graph: &Graph, node: &EntityNode) -> Result<()> {
    let labels_json = serde_json::to_string(&node.labels)?;
    let attributes_json = node.attributes.to_string();

    let q = query(
        "MERGE (n:EntityNode {uuid: $uuid}) \
         SET n.name = $name, \
             n.group_id = $group_id, \
             n.labels_json = $labels_json, \
             n.summary = $summary, \
             n.attributes = $attributes, \
             n.created_at = $created_at, \
             n.name_embedding = $name_embedding",
    )
    .param("uuid", node.uuid.to_string())
    .param("name", node.name.clone())
    .param("group_id", node.group_id.clone())
    .param("labels_json", labels_json)
    .param("summary", node.summary.clone())
    .param("attributes", attributes_json)
    .param("created_at", node.created_at.to_rfc3339())
    .param("name_embedding", opt_embedding_param(&node.name_embedding));

    graph.run(q).await.map_err(driver_err)
}

pub(super) async fn get_entity_node(graph: &Graph, uuid: &Uuid) -> Result<Option<EntityNode>> {
    let mut stream = graph
        .execute(
            query(
                "MATCH (n:EntityNode {uuid: $uuid}) \
                 RETURN n.uuid AS uuid, \
                        n.name AS name, \
                        n.group_id AS group_id, \
                        n.labels_json AS labels_json, \
                        n.summary AS summary, \
                        n.name_embedding AS name_embedding, \
                        n.attributes AS attributes, \
                        n.created_at AS created_at",
            )
            .param("uuid", uuid.to_string()),
        )
        .await
        .map_err(driver_err)?;

    if let Some(row) = stream.next().await.map_err(driver_err)? {
        Ok(Some(row_to_entity_node(&row)?))
    } else {
        Ok(None)
    }
}

pub(super) async fn delete_entity_node(graph: &Graph, uuid: &Uuid) -> Result<()> {
    graph
        .run(
            query("MATCH (n:EntityNode {uuid: $uuid}) DETACH DELETE n")
                .param("uuid", uuid.to_string()),
        )
        .await
        .map_err(driver_err)
}

// ── EpisodicNode ──────────────────────────────────────────────────────────────

fn episode_type_to_str(et: &EpisodeType) -> &'static str {
    match et {
        EpisodeType::Message => "Message",
        EpisodeType::Json => "Json",
        EpisodeType::Text => "Text",
    }
}

fn str_to_episode_type(s: &str) -> Result<EpisodeType> {
    match s {
        "Message" => Ok(EpisodeType::Message),
        "Json" => Ok(EpisodeType::Json),
        "Text" => Ok(EpisodeType::Text),
        other => Err(crate::errors::GraphitiError::Driver(format!(
            "unknown EpisodeType: {other}"
        ))),
    }
}

fn row_to_episodic_node(row: &neo4rs::Row) -> Result<EpisodicNode> {
    let uuid = parse_uuid(&row.get::<String>("uuid").map_err(driver_err)?)?;
    let name = row.get::<String>("name").map_err(driver_err)?;
    let group_id = row.get::<String>("group_id").map_err(driver_err)?;
    let labels_json = row.get::<String>("labels_json").map_err(driver_err)?;
    let labels: Vec<String> = serde_json::from_str(&labels_json)?;
    let created_at = parse_dt(&row.get::<String>("created_at").map_err(driver_err)?)?;
    let source_str = row.get::<String>("source").map_err(driver_err)?;
    let source = str_to_episode_type(&source_str)?;
    let source_description = row.get::<String>("source_description").map_err(driver_err)?;
    let content = row.get::<String>("content").map_err(driver_err)?;
    let valid_at = parse_dt(&row.get::<String>("valid_at").map_err(driver_err)?)?;
    let entity_edges_json = row.get::<String>("entity_edges_json").map_err(driver_err)?;
    let entity_edges: Vec<String> = serde_json::from_str(&entity_edges_json)?;

    Ok(EpisodicNode {
        uuid,
        name,
        group_id,
        labels,
        created_at,
        source,
        source_description,
        content,
        valid_at,
        entity_edges,
    })
}

pub(super) async fn save_episodic_node(graph: &Graph, node: &EpisodicNode) -> Result<()> {
    let labels_json = serde_json::to_string(&node.labels)?;
    let entity_edges_json = serde_json::to_string(&node.entity_edges)?;

    let q = query(
        "MERGE (n:EpisodicNode {uuid: $uuid}) \
         SET n.name = $name, \
             n.group_id = $group_id, \
             n.labels_json = $labels_json, \
             n.created_at = $created_at, \
             n.source = $source, \
             n.source_description = $source_description, \
             n.content = $content, \
             n.valid_at = $valid_at, \
             n.entity_edges_json = $entity_edges_json",
    )
    .param("uuid", node.uuid.to_string())
    .param("name", node.name.clone())
    .param("group_id", node.group_id.clone())
    .param("labels_json", labels_json)
    .param("created_at", node.created_at.to_rfc3339())
    .param("source", episode_type_to_str(&node.source))
    .param("source_description", node.source_description.clone())
    .param("content", node.content.clone())
    .param("valid_at", dt_param(node.valid_at))
    .param("entity_edges_json", entity_edges_json);

    graph.run(q).await.map_err(driver_err)
}

pub(super) async fn get_episodic_node(
    graph: &Graph,
    uuid: &Uuid,
) -> Result<Option<EpisodicNode>> {
    let mut stream = graph
        .execute(
            query(
                "MATCH (n:EpisodicNode {uuid: $uuid}) \
                 RETURN n.uuid AS uuid, \
                        n.name AS name, \
                        n.group_id AS group_id, \
                        n.labels_json AS labels_json, \
                        n.created_at AS created_at, \
                        n.source AS source, \
                        n.source_description AS source_description, \
                        n.content AS content, \
                        n.valid_at AS valid_at, \
                        n.entity_edges_json AS entity_edges_json",
            )
            .param("uuid", uuid.to_string()),
        )
        .await
        .map_err(driver_err)?;

    if let Some(row) = stream.next().await.map_err(driver_err)? {
        Ok(Some(row_to_episodic_node(&row)?))
    } else {
        Ok(None)
    }
}

pub(super) async fn delete_episodic_node(graph: &Graph, uuid: &Uuid) -> Result<()> {
    graph
        .run(
            query("MATCH (n:EpisodicNode {uuid: $uuid}) DETACH DELETE n")
                .param("uuid", uuid.to_string()),
        )
        .await
        .map_err(driver_err)
}

/// Return **all** episodic nodes that belong to `group_id`.
///
/// # Warning – unbounded result set
///
/// This function issues a Cypher query with **no `LIMIT` clause**, so the
/// returned `Vec` will contain every `EpisodicNode` stored under the given
/// `group_id`.  For small or medium groups this is fine, but for large groups
/// (thousands of episodes) the call will:
///
/// * consume significant Neo4j memory and I/O,
/// * transfer a large payload over the driver connection, and
/// * allocate a proportionally large `Vec` on the Rust heap.
///
/// Callers that only need a recent or size-bounded window of episodes should
/// use a more targeted query (e.g. with `ORDER BY n.created_at DESC LIMIT N`)
/// rather than filtering the returned `Vec` in application code.
pub(super) async fn list_episodic_nodes(
    graph: &Graph,
    group_id: &str,
) -> Result<Vec<EpisodicNode>> {
    let mut stream = graph
        .execute(
            query(
                "MATCH (n:EpisodicNode) WHERE n.group_id = $group_id \
                 RETURN n.uuid AS uuid, \
                        n.name AS name, \
                        n.group_id AS group_id, \
                        n.labels_json AS labels_json, \
                        n.created_at AS created_at, \
                        n.source AS source, \
                        n.source_description AS source_description, \
                        n.content AS content, \
                        n.valid_at AS valid_at, \
                        n.entity_edges_json AS entity_edges_json",
            )
            .param("group_id", group_id.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut nodes = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        nodes.push(row_to_episodic_node(&row)?);
    }
    Ok(nodes)
}

/// Return **all** entity nodes that belong to `group_id`.
pub(super) async fn list_entity_nodes(
    graph: &Graph,
    group_id: &str,
) -> Result<Vec<EntityNode>> {
    let mut stream = graph
        .execute(
            query(
                "MATCH (n:EntityNode) WHERE n.group_id = $group_id \
                 RETURN n.uuid AS uuid, \
                        n.name AS name, \
                        n.group_id AS group_id, \
                        n.labels_json AS labels_json, \
                        n.summary AS summary, \
                        n.name_embedding AS name_embedding, \
                        n.attributes AS attributes, \
                        n.created_at AS created_at",
            )
            .param("group_id", group_id.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut nodes = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        nodes.push(row_to_entity_node(&row)?);
    }
    Ok(nodes)
}

// ── CommunityNode ─────────────────────────────────────────────────────────────

pub(super) async fn save_community_node(graph: &Graph, node: &CommunityNode) -> Result<()> {
    let q = query(
        "MERGE (n:CommunityNode {uuid: $uuid}) \
         SET n.name = $name, \
             n.summary = $summary, \
             n.created_at = $created_at, \
             n.name_embedding = $name_embedding",
    )
    .param("uuid", node.uuid.to_string())
    .param("name", node.name.clone())
    .param("summary", node.summary.clone())
    .param("created_at", node.created_at.to_rfc3339())
    .param("name_embedding", opt_embedding_param(&node.name_embedding));

    graph.run(q).await.map_err(driver_err)
}

// ── Shared row-to-EntityNode (for search results) ─────────────────────────────

/// Re-usable extractor for rows that return individual EntityNode fields.
///
/// Used by both `get_entity_node` and search queries.
pub(super) fn row_to_entity_node_pub(row: &neo4rs::Row) -> Result<EntityNode> {
    row_to_entity_node(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn episode_type_roundtrips_through_string() {
        for et in [EpisodeType::Message, EpisodeType::Json, EpisodeType::Text] {
            let s = episode_type_to_str(&et);
            let recovered = str_to_episode_type(s).expect("roundtrip failed");
            assert_eq!(
                std::mem::discriminant(&recovered),
                std::mem::discriminant(&et)
            );
        }
    }

    #[test]
    fn str_to_episode_type_rejects_unknown() {
        assert!(str_to_episode_type("Unknown").is_err());
    }
}
