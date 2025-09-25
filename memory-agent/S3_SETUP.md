# S3 Storage Setup Guide

This guide helps you configure AWS S3 storage for the Memory Agent RAG system.

## Prerequisites

1. **AWS Account**: You need an AWS account with S3 access
2. **AWS Credentials**: Access key ID and secret access key
3. **S3 Bucket**: A bucket to store documents (will be created automatically if it doesn't exist)

## Configuration

### 1. Install Dependencies

```bash
pip install boto3 botocore
```

### 2. Set Environment Variables

Create a `.env` file in the `memory-agent` directory or set these environment variables:

```bash
# Required S3 Configuration
S3_BUCKET_NAME=your-memory-agent-documents
S3_REGION=us-east-1
S3_PREFIX=memory-agent-documents

# AWS Credentials
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
```

### 3. AWS Credentials Setup

#### Option A: Environment Variables (Recommended for development)
```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export S3_BUCKET_NAME=your-bucket-name
export S3_REGION=us-east-1
export S3_PREFIX=memory-agent-documents
```

#### Option B: AWS Credentials File
Create `~/.aws/credentials`:
```ini
[default]
aws_access_key_id = your_access_key_id
aws_secret_access_key = your_secret_access_key
```

Create `~/.aws/config`:
```ini
[default]
region = us-east-1
```

### 4. Create S3 Bucket (Optional)

The system will automatically create the bucket if it doesn't exist, but you can create it manually:

```bash
aws s3 mb s3://your-memory-agent-documents --region us-east-1
```

## Testing S3 Configuration

### Test Configuration
```python
from src.memory_agent.config import get_config
from src.memory_agent.s3_storage import create_s3_storage

config = get_config()
s3_storage = create_s3_storage(config)

if s3_storage:
    print("✅ S3 storage configured successfully")
else:
    print("❌ S3 storage configuration failed")
```

### Test Document Upload
```python
import asyncio

async def test_s3_upload():
    config = get_config()
    s3_storage = create_s3_storage(config)
    
    if not s3_storage:
        print("S3 storage not available")
        return
    
    # Test upload
    result = await s3_storage.store_document(
        content="This is a test document",
        document_id="test_doc_001",
        tenant_id="test_tenant",
        user_id="test_user",
        content_type="text/plain",
        filename="test.txt"
    )
    
    print(f"Upload result: {result}")

asyncio.run(test_s3_upload())
```

## Folder Structure

Documents are stored in S3 with this structure:
```
your-bucket/
└── memory-agent-documents/
    └── tenant_id/
        └── user_id/
            └── YYYY/
                └── MM/
                    └── document_id.ext
```

Example:
```
my-bucket/
└── memory-agent-documents/
    └── org1/
        └── user123/
            └── 2024/
                └── 01/
                    └── doc_1705312200_a1b2c3d4.pdf
```

## Troubleshooting

### Common Issues

1. **"S3 not available"**
   - Install boto3: `pip install boto3`
   - Check Python environment

2. **"S3 bucket name not configured"**
   - Set `S3_BUCKET_NAME` environment variable
   - Check `.env` file or environment

3. **"AWS credentials not found"**
   - Set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`
   - Or configure AWS credentials file

4. **"Access denied"**
   - Check AWS credentials permissions
   - Ensure S3 bucket exists and is accessible
   - Verify IAM policies

5. **"Bucket not found"**
   - Check bucket name spelling
   - Ensure bucket exists in the specified region
   - Check AWS region configuration

### Debug Commands

```bash
# Check AWS credentials
aws sts get-caller-identity

# List S3 buckets
aws s3 ls

# Test S3 access
aws s3 ls s3://your-bucket-name
```

## Security Best Practices

1. **Use IAM Roles**: For production, use IAM roles instead of access keys
2. **Least Privilege**: Grant only necessary S3 permissions
3. **Bucket Policies**: Configure appropriate bucket policies
4. **Encryption**: Enable S3 server-side encryption
5. **Access Logging**: Enable S3 access logging for audit trails

## Production Setup

For production environments:

1. **Use IAM Roles** instead of access keys
2. **Enable S3 encryption** at rest
3. **Configure bucket policies** for security
4. **Set up CloudTrail** for audit logging
5. **Use VPC endpoints** for private access
6. **Enable versioning** for document history

## Cost Optimization

- **Lifecycle policies**: Automatically delete old documents
- **Storage classes**: Use appropriate storage classes (Standard, IA, Glacier)
- **Compression**: Compress documents before upload
- **Monitoring**: Use CloudWatch to monitor costs
