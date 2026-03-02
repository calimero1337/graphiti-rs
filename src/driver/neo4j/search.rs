//! Search operations for the Neo4j driver.
//!
//! - `search_entity_nodes_by_name` — text CONTAINS match on node names
//! - `search_entity_edges_by_fact` — vector similarity search via Neo4j vector index
//! - `bm25_search_edges` — fulltext BM25 search via Neo4j fulltext index

use neo4rs::{query, Graph};

use crate::edges::EntityEdge;
use crate::errors::Result;
use crate::nodes::EntityNode;

use super::convert::driver_err;
use super::edges::row_to_entity_edge;
use super::nodes::row_to_entity_node_pub;

/// Cypher fragment returning all EntityNode fields; used in search queries.
const ENTITY_NODE_RETURN: &str = "RETURN n.uuid AS uuid, \
       n.name AS name, \
       n.group_id AS group_id, \
       n.labels_json AS labels_json, \
       n.summary AS summary, \
       n.name_embedding AS name_embedding, \
       n.attributes AS attributes, \
       n.created_at AS created_at";

/// Cypher fragment returning all EntityEdge fields from relationship `r`.
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

/// Text search for `EntityNode`s whose name contains `query_str` (case-insensitive).
pub(super) async fn search_entity_nodes_by_name(
    graph: &Graph,
    query_str: &str,
    group_id: &str,
    limit: usize,
) -> Result<Vec<EntityNode>> {
    // Use a safe integer limit formatted into the query (not user input).
    let cypher = format!(
        "MATCH (n:EntityNode) \
         WHERE n.group_id = $group_id \
           AND toLower(n.name) CONTAINS toLower($query) \
         {} \
         LIMIT {}",
        ENTITY_NODE_RETURN, limit
    );
    let mut stream = graph
        .execute(
            query(&cypher)
                .param("group_id", group_id.to_string())
                .param("query", query_str.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut nodes = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        nodes.push(row_to_entity_node_pub(&row)?);
    }
    Ok(nodes)
}

/// Vector similarity search for `EntityNode`s by `name_embedding`.
///
/// Uses the Neo4j vector index `entity_node_name_embedding_index` (created by
/// [`super::schema::build_indices`]).
pub(super) async fn search_entity_nodes_by_embedding(
    graph: &Graph,
    embedding: &[f32],
    group_id: &str,
    limit: usize,
) -> Result<Vec<EntityNode>> {
    let embedding_vec: Vec<f32> = embedding.to_vec();
    let cypher = format!(
        "CALL db.index.vector.queryNodes(\
             'entity_node_name_embedding_index', {}, $embedding) \
         YIELD node AS n, score \
         WHERE n.group_id = $group_id \
         {} \
         ORDER BY score DESC",
        limit, ENTITY_NODE_RETURN
    );
    let mut stream = graph
        .execute(
            query(&cypher)
                .param("embedding", embedding_vec)
                .param("group_id", group_id.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut nodes = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        nodes.push(row_to_entity_node_pub(&row)?);
    }
    Ok(nodes)
}

/// Vector similarity search for `EntityEdge`s by `fact_embedding`.
///
/// Uses the Neo4j vector index `entity_edge_fact_embedding_index` (created by
/// [`super::schema::build_indices`]).  Requires Neo4j 5.x with relationship
/// vector index support.
pub(super) async fn search_entity_edges_by_fact(
    graph: &Graph,
    embedding: &[f32],
    group_id: &str,
    limit: usize,
) -> Result<Vec<EntityEdge>> {
    let embedding_vec: Vec<f32> = embedding.to_vec();
    let cypher = format!(
        "CALL db.index.vector.queryRelationships(\
             'entity_edge_fact_embedding_index', {}, $embedding) \
         YIELD relationship AS r, score \
         WHERE r.group_id = $group_id \
         {} \
         ORDER BY score DESC",
        limit, ENTITY_EDGE_RETURN
    );
    let mut stream = graph
        .execute(
            query(&cypher)
                .param("embedding", embedding_vec)
                .param("group_id", group_id.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut edges = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        edges.push(row_to_entity_edge(&row)?);
    }
    Ok(edges)
}

/// BM25 fulltext search for `EntityEdge`s by fact text.
///
/// Uses the Neo4j fulltext index `entity_edge_fact_fulltext_index` (created by
/// [`super::schema::build_indices`]).
pub(super) async fn bm25_search_edges(
    graph: &Graph,
    query_str: &str,
    group_id: &str,
    limit: usize,
) -> Result<Vec<EntityEdge>> {
    let cypher = format!(
        "CALL db.index.fulltext.queryRelationships(\
             'entity_edge_fact_fulltext_index', $query, {{limit: {}}}) \
         YIELD relationship AS r, score \
         WHERE r.group_id = $group_id \
         {} \
         ORDER BY score DESC",
        limit, ENTITY_EDGE_RETURN
    );
    let mut stream = graph
        .execute(
            query(&cypher)
                .param("query", query_str.to_string())
                .param("group_id", group_id.to_string()),
        )
        .await
        .map_err(driver_err)?;

    let mut edges = Vec::new();
    while let Some(row) = stream.next().await.map_err(driver_err)? {
        edges.push(row_to_entity_edge(&row)?);
    }
    Ok(edges)
}
