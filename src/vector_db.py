"""Vector database for query similarity search using PostgreSQL + pgvector."""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
from psycopg2.extensions import register_adapter, AsIs


def addapt_numpy_float64(numpy_float64):
    """Adapter for numpy float64 to PostgreSQL."""
    return AsIs(numpy_float64)

def addapt_numpy_int64(numpy_int64):
    """Adapter for numpy int64 to PostgreSQL."""
    return AsIs(numpy_int64)

register_adapter(np.float64, addapt_numpy_float64)
register_adapter(np.int64, addapt_numpy_int64)


class QueryVectorDB:
    """Vector database for storing and searching similar queries using PostgreSQL."""

    def __init__(
        self,
        host: str = None,
        port: int = None,
        database: str = None,
        user: str = None,
        password: str = None
    ):
        """Initialize vector database connection.

        Args:
            host: Database host (will use env var POSTGRES_HOST if not provided)
            port: Database port (will use env var POSTGRES_PORT if not provided)
            database: Database name (will use env var POSTGRES_DB if not provided)
            user: Database user (will use env var POSTGRES_USER if not provided)
            password: Database password (will use env var POSTGRES_PASSWORD if not provided)
        """
        if host is None:
            host = os.getenv("POSTGRES_HOST", "db.ezrhboaaipclzspfffxk.supabase.co")
        if port is None:
            port = int(os.getenv("POSTGRES_PORT", "5432"))
        if database is None:
            database = os.getenv("POSTGRES_DB", "postgres")
        if user is None:
            user = os.getenv("POSTGRES_USER", "postgres")
        if password is None:
            password = os.getenv("POSTGRES_PASSWORD", "")

        self.connection_params = {
            "host": host,
            "port": port,
            "database": database,
            "user": user,
            "password": password
        }

        self.conn = None
        self._connect()
        self._initialize_database()

    def _connect(self):
        """Establish database connection."""
        try:
            self.conn = psycopg2.connect(**self.connection_params)
            self.conn.autocommit = False
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {e}")

    def _initialize_database(self):
        """Initialize database schema with pgvector extension."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS coding_queries (
                        id TEXT PRIMARY KEY,
                        prompt TEXT NOT NULL,
                        embedding vector(384),
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS coding_queries_embedding_idx
                    ON coding_queries
                    USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)

                cur.execute("""
                    CREATE INDEX IF NOT EXISTS coding_queries_metadata_idx
                    ON coding_queries
                    USING gin (metadata);
                """)

                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to initialize database: {e}")

    def add_query(
        self,
        query_id: str,
        prompt: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> None:
        """Add a query to the database.

        Args:
            query_id: Unique identifier for the query
            prompt: The original prompt text
            embedding: Query embedding vector (384-dimensional)
            metadata: Additional metadata (model_used, success, latency, etc.)
        """
        try:
            with self.conn.cursor() as cur:
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                cur.execute("""
                    INSERT INTO coding_queries (id, prompt, embedding, metadata)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (id)
                    DO UPDATE SET
                        prompt = EXCLUDED.prompt,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP;
                """, (query_id, prompt, embedding_list, json.dumps(metadata)))

                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add query: {e}")

    def add_queries_batch(
        self,
        query_ids: List[str],
        prompts: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """Add multiple queries in batch.

        Args:
            query_ids: List of unique identifiers
            prompts: List of prompt texts
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
        """
        try:
            with self.conn.cursor() as cur:
                data = [
                    (
                        qid,
                        prompt,
                        emb.tolist() if isinstance(emb, np.ndarray) else emb,
                        json.dumps(meta)
                    )
                    for qid, prompt, emb, meta in zip(query_ids, prompts, embeddings, metadatas)
                ]

                execute_values(
                    cur,
                    """
                    INSERT INTO coding_queries (id, prompt, embedding, metadata)
                    VALUES %s
                    ON CONFLICT (id)
                    DO UPDATE SET
                        prompt = EXCLUDED.prompt,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        created_at = CURRENT_TIMESTAMP;
                    """,
                    data
                )

                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to add queries batch: {e}")

    def find_similar(
        self,
        embedding: np.ndarray,
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[str], List[float], List[Dict[str, Any]]]:
        """Find k most similar queries using cosine similarity.

        Args:
            embedding: Query embedding to search for
            k: Number of similar queries to return
            filter_metadata: Optional metadata filters

        Returns:
            Tuple of (ids, distances, metadatas)
        """
        try:
            with self.conn.cursor() as cur:
                embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

                query = """
                    SELECT id, embedding <=> %s::vector AS distance, metadata
                    FROM coding_queries
                """
                params = [embedding_list]

                if filter_metadata:
                    conditions = []
                    for key, value in filter_metadata.items():
                        conditions.append(f"metadata->>{key} = %s")
                        params.append(str(value))
                    query += " WHERE " + " AND ".join(conditions)

                query += " ORDER BY distance LIMIT %s;"
                params.append(k)

                cur.execute(query, params)
                results = cur.fetchall()

                if not results:
                    return [], [], []

                ids = [row[0] for row in results]
                distances = [float(row[1]) for row in results]
                metadatas = [row[2] for row in results]

                return ids, distances, metadatas
        except Exception as e:
            raise RuntimeError(f"Failed to find similar queries: {e}")

    def calculate_local_success_rate(
        self,
        embedding: np.ndarray,
        k: int = 10
    ) -> float:
        """Calculate success rate of local model on similar queries.

        Args:
            embedding: Query embedding
            k: Number of similar queries to consider

        Returns:
            Success rate (0.0 to 1.0), or 0.5 if no data
        """
        ids, distances, metadatas = self.find_similar(embedding, k=k)

        if not metadatas:
            return 0.5

        local_queries = [m for m in metadatas if m.get('model_used') == 'local']

        if not local_queries:
            return 0.5

        successes = sum(1 for m in local_queries if m.get('success', False))
        return successes / len(local_queries)

    def calculate_similarity_to_failures(
        self,
        embedding: np.ndarray,
        k: int = 10
    ) -> float:
        """Calculate similarity to queries where local model failed.

        Args:
            embedding: Query embedding
            k: Number of similar queries to consider

        Returns:
            Similarity score (0.0 to 1.0), higher means more similar to failures
        """
        ids, distances, metadatas = self.find_similar(embedding, k=k)

        if not metadatas:
            return 0.0

        failures = [
            (d, m) for d, m in zip(distances, metadatas)
            if m.get('model_used') == 'local' and not m.get('success', True)
        ]

        if not failures:
            return 0.0

        avg_distance = sum(d for d, _ in failures) / len(failures)

        similarity = 1.0 - avg_distance

        return max(0.0, min(1.0, similarity))

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics.

        Returns:
            Statistics dictionary
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM coding_queries;")
                count = cur.fetchone()[0]

                if count == 0:
                    return {
                        "total_queries": 0,
                        "local_queries": 0,
                        "cloud_queries": 0,
                        "success_rate": 0.0,
                        "sample_size": 0
                    }

                sample_size = min(count, 100)
                cur.execute("""
                    SELECT metadata
                    FROM coding_queries
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (sample_size,))

                metadatas = [row[0] for row in cur.fetchall()]

                local_count = sum(1 for m in metadatas if m.get('model_used') == 'local')
                cloud_count = sum(1 for m in metadatas if m.get('model_used') == 'cloud')
                success_count = sum(1 for m in metadatas if m.get('success', False))

                return {
                    "total_queries": count,
                    "local_queries": local_count,
                    "cloud_queries": cloud_count,
                    "success_rate": success_count / sample_size if sample_size > 0 else 0.0,
                    "sample_size": sample_size
                }
        except Exception as e:
            raise RuntimeError(f"Failed to get statistics: {e}")

    def clear(self) -> None:
        """Clear all data from the database."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM coding_queries;")
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise RuntimeError(f"Failed to clear database: {e}")

    def export_to_json(self, output_path: str) -> None:
        """Export database to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM coding_queries;")
                count = cur.fetchone()[0]

                if count == 0:
                    print("No data to export")
                    return

                cur.execute("""
                    SELECT id, prompt, embedding, metadata
                    FROM coding_queries
                    ORDER BY created_at;
                """)

                data = {"queries": []}

                for row in cur.fetchall():
                    data["queries"].append({
                        "id": row[0],
                        "prompt": row[1],
                        "embedding": row[2],
                        "metadata": row[3]
                    })

                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    json.dump(data, f, indent=2)

                print(f"Exported {count} queries to {output_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to export to JSON: {e}")

    def import_from_json(self, input_path: str) -> None:
        """Import data from JSON file.

        Args:
            input_path: Path to input JSON file
        """
        with open(input_path, 'r') as f:
            data = json.load(f)

        queries = data.get('queries', [])
        if not queries:
            print("No queries to import")
            return

        ids = [q['id'] for q in queries]
        prompts = [q['prompt'] for q in queries]
        embeddings = [np.array(q['embedding']) for q in queries]
        metadatas = [q['metadata'] for q in queries]

        self.add_queries_batch(ids, prompts, embeddings, metadatas)
        print(f"Imported {len(queries)} queries from {input_path}")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
