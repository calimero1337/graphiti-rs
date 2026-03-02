//! Schema and index management for the Neo4j driver.
//!
//! Creates vector and fulltext indexes required by the graphiti-rs graph driver.
//!
//! ## Indexes created
//!
//! | Name | Type | Target | Property |
//! |------|------|--------|----------|
//! | `entity_node_name_embedding_index` | Vector | `:EntityNode` | `name_embedding` |
//! | `entity_node_uuid_index` | Range | `:EntityNode` | `uuid` |
//! | `entity_edge_fact_fulltext_index` | Fulltext | `RELATES_TO` | `fact` |
//! | `entity_edge_fact_embedding_index` | Vector | `RELATES_TO` | `fact_embedding` |

use neo4rs::{query, Graph};

use crate::errors::Result;

use super::convert::driver_err;

/// Vector embedding dimensions (must match the embedder model output size).
///
/// 1536 matches OpenAI `text-embedding-3-small` / `text-embedding-ada-002`.
/// Override if using a different model.
const VECTOR_DIMENSIONS: usize = 1536;

/// Create all required indexes and constraints in Neo4j.
///
/// Uses `IF NOT EXISTS` so this is safe to call multiple times.
pub(super) async fn build_indices(graph: &Graph) -> Result<()> {
    // Uniqueness constraint on EntityNode.uuid (also creates a lookup index).
    run_ddl(
        graph,
        "CREATE CONSTRAINT entity_node_uuid_unique IF NOT EXISTS \
         FOR (n:EntityNode) REQUIRE n.uuid IS UNIQUE",
    )
    .await?;

    // Uniqueness constraint on EpisodicNode.uuid.
    run_ddl(
        graph,
        "CREATE CONSTRAINT episodic_node_uuid_unique IF NOT EXISTS \
         FOR (n:EpisodicNode) REQUIRE n.uuid IS UNIQUE",
    )
    .await?;

    // Uniqueness constraint on CommunityNode.uuid.
    run_ddl(
        graph,
        "CREATE CONSTRAINT community_node_uuid_unique IF NOT EXISTS \
         FOR (n:CommunityNode) REQUIRE n.uuid IS UNIQUE",
    )
    .await?;

    // Vector index for entity node name embeddings.
    let vector_node_ddl = format!(
        "CREATE VECTOR INDEX entity_node_name_embedding_index IF NOT EXISTS \
         FOR (n:EntityNode) ON (n.name_embedding) \
         OPTIONS {{indexConfig: {{\
           `vector.dimensions`: {VECTOR_DIMENSIONS}, \
           `vector.similarity_function`: 'cosine'\
         }}}}"
    );
    run_ddl(graph, &vector_node_ddl).await?;

    // Fulltext index on RELATES_TO fact for BM25 search.
    run_ddl(
        graph,
        "CREATE FULLTEXT INDEX entity_edge_fact_fulltext_index IF NOT EXISTS \
         FOR ()-[r:RELATES_TO]-() ON EACH [r.fact]",
    )
    .await?;

    // Vector index for entity edge fact embeddings (Neo4j 5.18+).
    let vector_edge_ddl = format!(
        "CREATE VECTOR INDEX entity_edge_fact_embedding_index IF NOT EXISTS \
         FOR ()-[r:RELATES_TO]-() ON (r.fact_embedding) \
         OPTIONS {{indexConfig: {{\
           `vector.dimensions`: {VECTOR_DIMENSIONS}, \
           `vector.similarity_function`: 'cosine'\
         }}}}"
    );
    run_ddl(graph, &vector_edge_ddl).await?;

    Ok(())
}

async fn run_ddl(graph: &Graph, cypher: &str) -> Result<()> {
    graph.run(query(cypher)).await.map_err(driver_err)
}

#[cfg(test)]
mod tests {
    #[test]
    fn vector_dimensions_is_positive() {
        assert!(super::VECTOR_DIMENSIONS > 0);
    }
}
