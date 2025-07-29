"""
S3 Vector Database implementation for VSB (Vector Search Benchmarking).

This module provides an S3 vector database client that follows the VSB interface
pattern. It uses S3 Vectors for storage and retrieval of pre-computed embeddings.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from ...vsb_types import DistanceMetric, SearchRequest

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:
    raise ImportError("boto3 is required for S3 vector database. Install with: pip install boto3")

from ..base import DB, Namespace


logger = logging.getLogger(__name__)


class S3VecDB(DB):
    """S3 Vector Database implementation."""

    def __init__(self,
                 dimensions: int,
                 metric: DistanceMetric,
                 name: str,
                 **kwargs):
        """
        Initialize S3 Vector Database.

        Args:
            **kwargs: Arguments passed from VSB command line, including:
                s3vec_region: AWS region for S3 Vectors
                s3vec_bucket_name: S3 vector bucket name
                s3vec_index_name: S3 vector index name
                s3vec_aws_access_key_id: AWS access key ID (optional)
                s3vec_aws_secret_access_key: AWS secret access key (optional)
                s3vec_aws_session_token: AWS session token (optional)
        """
        self.dimensions = dimensions
        self.metric = metric
        # Extract S3Vec specific arguments from kwargs
        region_name = kwargs.get('s3vec_region', 'us-west-2')
        bucket_name = kwargs.get('s3vec_bucket_name')
        index_name = kwargs.get('s3vec_index_name')
        aws_access_key_id = kwargs.get('s3vec_aws_access_key_id')
        aws_secret_access_key = kwargs.get('s3vec_aws_secret_access_key')
        aws_session_token = kwargs.get('s3vec_aws_session_token')

        self.region_name = region_name
        self.vector_bucket_name = bucket_name or f"{name}"
        # Use index name if provided, otherwise use bucket name as base but add suffix
        if index_name:
            self.s3_index_name = index_name
        elif bucket_name:
            self.s3_index_name = bucket_name  # Use bucket name as index name
        else:
            self.s3_index_name = f"{name}"  # Different from bucket name

        # Configure AWS session
        session_config = {"region_name": region_name}
        if aws_access_key_id:
            session_config["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_config["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_config["aws_session_token"] = aws_session_token

        try:
            self.session = boto3.Session(**session_config)
            self.s3vectors_client = self.session.client('s3vectors')
            logger.info(f"Initialized S3 Vector DB with bucket: {self.vector_bucket_name}, S3 index: {self.s3_index_name}")
        except NoCredentialsError:
            raise ValueError("AWS credentials not found. Please configure AWS credentials.")
        except Exception as e:
            raise ValueError(f"Failed to initialize AWS clients: {e}")

        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to AWS S3 Vectors service."""
        try:
            # Test S3 Vectors connection
            self.s3vectors_client.list_vector_buckets()
            logger.debug("S3 Vectors connection successful")
        except ClientError as e:
            if "UnauthorizedOperation" not in str(e):
                raise
            logger.debug("S3 Vectors service accessible (permission check passed)")
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            raise ValueError(f"Cannot connect to S3 Vectors service: {e}")

    def get_namespace(self, name: str) -> Namespace:
        """Get a namespace (index) for the given name."""
        return S3VecNamespace(
            client=self,
            name=name,
            vector_bucket_name=self.vector_bucket_name,
            s3_index_name=f"{self.s3_index_name}{f'-{name.strip()}' if name.strip() else ''}",
            dimensions=self.dimensions,
            metric=self.metric
        )

    def get_batch_size(self, sample_record) -> int:
        """Return the preferred batch size for populate operations."""
        # S3 Vectors has a 2MB request limit
        vector_size_bytes = self.dimensions * 4  # 4 bytes per float32
        max_vectors = (2 * 1024 * 1024) // (vector_size_bytes + 100)  # 100 bytes buffer for metadata
        return min(max_vectors, 500)  # Cap at 500 for safety

    def initialize_populate(self):
        """Initialize the database for population phase."""
        logger.info("Starting S3 Vector DB population initialization")
        # Create vector bucket if it doesn't exist using regular S3 client
        try:
            # Use regular S3 client for bucket creation, as per the example
            s3_client = self.session.client('s3')
            s3_client.create_bucket(Bucket=self.vector_bucket_name)
            logger.info(f"Created vector bucket: {self.vector_bucket_name}")
        except ClientError as e:
            if "BucketAlreadyOwnedByYou" in str(e) or "BucketAlreadyExists" in str(e):
                logger.info(f"Vector bucket already exists: {self.vector_bucket_name}")
            else:
                logger.error(f"Failed to create vector bucket: {e}")
                raise

    def finalize_populate(self):
        """Finalize the population phase."""
        logger.info("S3 Vector DB population finalized")
        # S3 Vectors doesn't require explicit index building like some databases
        # Vectors are immediately searchable after insertion


class S3VecNamespace(Namespace):
    """S3 Vector Namespace implementation."""

    def __init__(self,
                 client: S3VecDB,
                 name: str,
                 vector_bucket_name: str,
                 s3_index_name: str,
                 dimensions: int,
                 metric: DistanceMetric):
        """
        Initialize S3 Vector Namespace.

        Args:
            client: Parent S3VecDB client
            name: Namespace name
            vector_bucket_name: S3 vector bucket name
            s3_index_name: S3 vector index name
        """
        self.client = client
        self.name = name
        self.vector_bucket_name = vector_bucket_name
        self.s3_index_name = s3_index_name
        self.dimensions = dimensions
        self.metric = metric

        # Check if index exists and get its dimension
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        """Check if the vector index exists and get its dimension."""
        try:
            # List indexes in the vector bucket to check if our index exists
            response = self.client.s3vectors_client.list_indexes(
                vectorBucketName=self.vector_bucket_name  # Capital V as per example
            )

            # Look for our specific index
            for index in response['indexes']:
                if index.get('indexName') == self.s3_index_name:
                    logger.debug(f"Index {self.s3_index_name} already exists")
                    return


        except Exception as e:
            logger.warning(f"Could not list indexes: {e}")
            # Continue without error - we'll try to create the index when needed

        # Index doesn't exist, we'll create it when we know the vector dimension
        logger.info(f"Index {self.s3_index_name} does not exist, creating it with metric {self.metric.name}")

        self.client.s3vectors_client.create_index(
            vectorBucketName=self.vector_bucket_name,  # Capital V as per example
            indexName=self.s3_index_name,
            dimension=self.dimensions,  # 'Dimension', not 'vectorDimensions'
            distanceMetric=self.metric.name.lower(),  # 'DistanceMetric', not 'indexMetric'
            dataType="float32"  # Required parameter
            )
        logger.info(f"Created vector index: {self.s3_index_name} with dimension {self.dimensions}")


    def upsert_batch(self, records):
        """
        Upsert a batch of vector records.

        Args:
            records: List of Record objects with id, values, and optional metadata
        """
        if not records:
            return

        logger.debug(f"Upserting batch of {len(records)} records to {self.s3_index_name}")

        # Convert VSB records to S3 Vectors format
        vectors = []
        for record in records:
            # VSB Record object has: id, values, metadata
            record_id = record.id
            record_vector = record.values  # Vector data is in 'values' attribute
            record_metadata = record.metadata

            # VSB always provides pre-computed vectors
            if record_vector is None:
                logger.warning(f"Record {record_id} has no vector data, skipping")
                continue

            # Build metadata
            metadata = {"id": record_id}
            if record_metadata:
                metadata.update(record_metadata)

            # Format vector for S3 Vectors
            s3_vector = {
                "key": record_id,
                "data": {"float32": [float(x) for x in record_vector]},  # Convert numpy types to Python float
                "metadata": metadata
            }
            vectors.append(s3_vector)

        if not vectors:
            logger.warning("No valid vectors to upsert")
            return

        # Batch upsert to S3 Vectors
        try:
            self.client.s3vectors_client.put_vectors(
                vectorBucketName=self.vector_bucket_name,  # lowercase 'v'
                indexName=self.s3_index_name,
                vectors=vectors
            )
            logger.debug(f"Successfully upserted {len(vectors)} vectors")
        except Exception as e:
            logger.error(f"Failed to upsert vectors: {e}")
            raise

    def search(
        self,
       request: SearchRequest
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query vector
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of result dictionaries with 'id', 'score', and 'metadata'
        """
        vector = request.values
        top_k = request.top_k
        filter = request.filter
        logger.debug(f"Searching {self.s3_index_name} for top {top_k} results")

        try:
            # Build query parameters
            query_params = {
                "vectorBucketName": self.vector_bucket_name,  # lowercase 'v'
                "indexName": self.s3_index_name,
                "queryVector": {"float32": vector},
                "topK": 30,
                "returnDistance": True,
                "returnMetadata": True
            }

            # Add filters if provided
            if filter:
                query_params["filter"] = filter

            # Execute search
            response = self.client.s3vectors_client.query_vectors(**query_params)
            # Convert results to VSB format
            results = []
            for vector_result in response.get("vectors", []):
                # Extract metadata
                metadata = vector_result.get("metadata", {})
                vector_id = metadata.get("id", vector_result.get("key"))

                # Create result dictionary
                results.append(vector_id)

            logger.debug(f"Found {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def insert_batch(self, records):
        """
        Insert a batch of vector records (alias for upsert_batch).

        Args:
            records: List of Record objects or dictionaries containing 'id', 'vector', and optional 'metadata'
        """
        return self.upsert_batch(records)

    def update_batch(self, records):
        """
        Update a batch of vector records (alias for upsert_batch).

        Args:
            records: List of Record objects or dictionaries containing 'id', 'vector', and optional 'metadata'
        """
        return self.upsert_batch(records)

    def delete_batch(self, vector_ids: List[str]):
        """
        Delete vectors by their IDs.

        Args:
            vector_ids: List of vector IDs to delete
        """
        logger.debug(f"Deleting {len(vector_ids)} vectors from {self.s3_index_name}")

        try:
            self.client.s3vectors_client.delete_vectors(
                vectorBucketName=self.vector_bucket_name,  # lowercase 'v'
                indexName=self.s3_index_name,
                vectorKeys=vector_ids
            )
            logger.debug(f"Successfully deleted {len(vector_ids)} vectors")
        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            raise

    def fetch_batch(self, vector_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch vectors by their IDs.

        Args:
            vector_ids: List of vector IDs to fetch

        Returns:
            List of record dictionaries with 'id', 'vector', and 'metadata'
        """
        logger.debug(f"Fetching {len(vector_ids)} vectors from {self.s3_index_name}")

        try:
            # S3 Vectors doesn't have a direct fetch API, so we'll use a workaround
            # This is a limitation of S3 Vectors - it's primarily designed for similarity search
            # For now, we'll return empty results and log a warning
            logger.warning("S3 Vectors does not support direct vector fetching by ID")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch vectors: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the namespace."""
        try:
            # Use list_indexes to get basic index information
            response = self.client.s3vectors_client.list_indexes(
                VectorBucketName=self.vector_bucket_name  # Capital V
            )

            # Find our specific index
            if 'IndexList' in response:
                for index in response['IndexList']:
                    if index.get('IndexName') == self.s3_index_name:
                        return {
                            "vector_count": "unknown",  # S3 Vectors doesn't provide count in list_indexes
                            "dimension": index.get('Dimension', 0),  # 'Dimension', not 'VectorDimensions'
                            "index_status": index.get('Status', 'unknown')
                        }

            # Index not found
            return {
                "vector_count": 0,
                "dimension": self._vector_dimension or 0,
                "index_status": "not_found"
            }

        except Exception as e:
            logger.warning(f"Failed to get stats: {e}")
            return {
                "vector_count": 0,
                "dimension": self._vector_dimension or 0,
                "index_status": "error"
            }