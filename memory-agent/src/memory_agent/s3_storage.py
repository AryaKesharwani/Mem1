"""
S3 Document Storage - AWS S3 integration for document storage.

This module provides S3-based document storage for the RAG system,
storing original documents in S3 while using Pinecone for vector search.
"""

import asyncio
import logging
import mimetypes
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import hashlib

# Try to import boto3, fall back gracefully if not available
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

from .config import MemoryConfig


class S3DocumentStorage:
    """
    S3-based document storage for RAG system.
    
    Stores original documents in S3 buckets with organized folder structure
    and provides metadata for Pinecone vector storage.
    """
    
    def __init__(self, config: MemoryConfig):
        """Initialize S3 document storage."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        if not S3_AVAILABLE:
            raise ImportError("boto3 not available. Install with: pip install boto3")
        
        # S3 configuration
        self.bucket_name = config.s3_bucket_name
        self.region = config.s3_region
        self.prefix = config.s3_prefix or "memory-agent-documents"
        
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                region_name=self.region,
                aws_access_key_id=config.aws_access_key_id,
                aws_secret_access_key=config.aws_secret_access_key
            )
            
            # Test connection and create bucket if needed
            self._ensure_bucket_exists()
            
            self.logger.info(f"S3DocumentStorage initialized: bucket={self.bucket_name}, region={self.region}")
            
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 client: {str(e)}")
            raise
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create if it doesn't."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            self.logger.debug(f"S3 bucket '{self.bucket_name}' exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.region == 'us-east-1':
                        # us-east-1 doesn't need LocationConstraint
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    self.logger.info(f"Created S3 bucket: {self.bucket_name}")
                except ClientError as create_error:
                    self.logger.error(f"Failed to create S3 bucket: {str(create_error)}")
                    raise
            else:
                self.logger.error(f"Error checking S3 bucket: {str(e)}")
                raise
    
    def _generate_s3_key(self, tenant_id: str, user_id: str, document_id: str, 
                        content_type: str, filename: Optional[str] = None) -> str:
        """Generate S3 object key for document storage."""
        # Create organized folder structure
        # Format: prefix/tenant_id/user_id/year/month/document_id.ext
        
        now = datetime.now(timezone.utc)
        year = now.strftime("%Y")
        month = now.strftime("%m")
        
        # Determine file extension from content type or filename
        if filename:
            ext = Path(filename).suffix
        else:
            ext = mimetypes.guess_extension(content_type) or '.txt'
        
        # Sanitize document_id for S3 key
        safe_doc_id = "".join(c for c in document_id if c.isalnum() or c in '-_.')
        
        s3_key = f"{self.prefix}/{tenant_id}/{user_id}/{year}/{month}/{safe_doc_id}{ext}"
        return s3_key
    
    async def store_document(
        self,
        content: Union[str, bytes],
        document_id: str,
        tenant_id: str,
        user_id: str,
        content_type: str = "text/plain",
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store document in S3.
        
        Args:
            content: Document content (text or bytes)
            document_id: Unique document identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            content_type: MIME type of the document
            filename: Original filename (optional)
            metadata: Additional metadata to store
            
        Returns:
            Dict with S3 storage information
        """
        try:
            # Convert string content to bytes if needed
            if isinstance(content, str):
                content_bytes = content.encode('utf-8')
            else:
                content_bytes = content
            
            # Generate S3 key
            s3_key = self._generate_s3_key(tenant_id, user_id, document_id, content_type, filename)
            
            # Prepare metadata for S3
            s3_metadata = {
                'document_id': document_id,
                'tenant_id': tenant_id,
                'user_id': user_id,
                'content_type': content_type,
                'upload_timestamp': datetime.now(timezone.utc).isoformat(),
                'content_length': str(len(content_bytes))
            }
            
            if filename:
                s3_metadata['original_filename'] = filename
            
            if metadata:
                # Add custom metadata (S3 metadata keys must be lowercase)
                for key, value in metadata.items():
                    safe_key = key.lower().replace(' ', '-')
                    s3_metadata[f'custom-{safe_key}'] = str(value)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=content_bytes,
                ContentType=content_type,
                Metadata=s3_metadata
            )
            
            # Generate S3 URL
            s3_url = f"s3://{self.bucket_name}/{s3_key}"
            
            # Calculate content hash for integrity
            content_hash = hashlib.sha256(content_bytes).hexdigest()
            
            result = {
                "status": "success",
                "document_id": document_id,
                "s3_key": s3_key,
                "s3_url": s3_url,
                "bucket": self.bucket_name,
                "content_type": content_type,
                "content_length": len(content_bytes),
                "content_hash": content_hash,
                "upload_timestamp": s3_metadata['upload_timestamp']
            }
            
            self.logger.info(f"Document stored in S3: {document_id} -> {s3_key}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to store document in S3: {str(e)}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e)
            }
    
    async def retrieve_document(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve document from S3.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            Dict with document content and metadata, or None if not found
        """
        try:
            # List objects to find the document (since we don't know the exact key)
            prefix_pattern = f"{self.prefix}/{tenant_id}/{user_id}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix_pattern
            )
            
            # Find the object with matching document_id in metadata
            target_key = None
            for obj in response.get('Contents', []):
                try:
                    # Get object metadata
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
                    metadata = head_response.get('Metadata', {})
                    if metadata.get('document_id') == document_id:
                        target_key = obj['Key']
                        break
                        
                except ClientError:
                    continue
            
            if not target_key:
                self.logger.warning(f"Document not found in S3: {document_id}")
                return None
            
            # Retrieve the object
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=target_key
            )
            
            content = response['Body'].read()
            metadata = response.get('Metadata', {})
            
            # Determine if content is text or binary
            content_type = metadata.get('content-type', 'text/plain')
            if content_type.startswith('text/'):
                content_str = content.decode('utf-8')
            else:
                content_str = None
            
            result = {
                "document_id": document_id,
                "content": content_str,
                "content_bytes": content,
                "content_type": content_type,
                "metadata": metadata,
                "s3_key": target_key,
                "content_length": len(content)
            }
            
            self.logger.info(f"Document retrieved from S3: {document_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve document from S3: {str(e)}")
            return None
    
    async def delete_document(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str
    ) -> bool:
        """
        Delete document from S3.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Find the document first
            document_info = await self.retrieve_document(document_id, tenant_id, user_id)
            if not document_info:
                return False
            
            s3_key = document_info['s3_key']
            
            # Delete from S3
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            self.logger.info(f"Document deleted from S3: {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete document from S3: {str(e)}")
            return False
    
    async def list_documents(
        self,
        tenant_id: str,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List documents for a tenant/user.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            limit: Maximum number of documents to return
            
        Returns:
            List of document metadata
        """
        try:
            prefix_pattern = f"{self.prefix}/{tenant_id}/{user_id}/"
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix_pattern,
                MaxKeys=limit
            )
            
            documents = []
            for obj in response.get('Contents', []):
                try:
                    # Get object metadata
                    head_response = self.s3_client.head_object(
                        Bucket=self.bucket_name,
                        Key=obj['Key']
                    )
                    
                    metadata = head_response.get('Metadata', {})
                    
                    document_info = {
                        "document_id": metadata.get('document_id', 'unknown'),
                        "s3_key": obj['Key'],
                        "content_type": metadata.get('content-type', 'unknown'),
                        "content_length": int(metadata.get('content-length', 0)),
                        "upload_timestamp": metadata.get('upload-timestamp', 'unknown'),
                        "last_modified": obj['LastModified'].isoformat(),
                        "size": obj['Size']
                    }
                    
                    documents.append(document_info)
                    
                except ClientError:
                    continue
            
            self.logger.info(f"Listed {len(documents)} documents for {tenant_id}/{user_id}")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {str(e)}")
            return []
    
    async def get_document_url(
        self,
        document_id: str,
        tenant_id: str,
        user_id: str,
        expiration: int = 3600
    ) -> Optional[str]:
        """
        Generate a presigned URL for document access.
        
        Args:
            document_id: Document identifier
            tenant_id: Tenant identifier
            user_id: User identifier
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            Presigned URL or None if document not found
        """
        try:
            # Find the document first
            document_info = await self.retrieve_document(document_id, tenant_id, user_id)
            if not document_info:
                return None
            
            s3_key = document_info['s3_key']
            
            # Generate presigned URL
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            
            self.logger.info(f"Generated presigned URL for document: {document_id}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to generate presigned URL: {str(e)}")
            return None


# Convenience function for creating S3 storage
def create_s3_storage(config: MemoryConfig) -> Optional[S3DocumentStorage]:
    """Create S3 document storage instance."""
    if not S3_AVAILABLE:
        logging.warning("S3 not available. Install boto3 to enable S3 document storage.")
        return None
    
    if not config.s3_bucket_name:
        logging.warning("S3 bucket name not configured. Set S3_BUCKET_NAME environment variable.")
        return None
    
    try:
        return S3DocumentStorage(config)
    except Exception as e:
        logging.error(f"Failed to create S3 storage: {str(e)}")
        return None
