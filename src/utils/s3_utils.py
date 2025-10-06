import boto3
import os
from botocore.exceptions import ClientError

s3 = boto3.client("s3")


def download_images_from_s3(s3_paths: list[str], local_dir: str = "/tmp"):
    downloaded_files = []
    for s3_path in s3_paths:
        bucket, key = s3_path.replace("s3://", "").split("/", 1)
        local_path = os.path.join(local_dir, os.path.basename(key))
        s3.download_file(bucket, key, local_path)
        downloaded_files.append(local_path)
    return downloaded_files


def upload_file_to_s3(local_path: str, s3_prefix: str = "scientific_reports") -> str:
    """
    Uploads a local file to S3 and returns the public URL.
    """
    bucket_name = os.getenv("S3_BUCKET_NAME")
    key = f"{s3_prefix}/{os.path.basename(local_path)}"

    try:
        s3.upload_file(local_path, bucket_name, key, ExtraArgs={"ACL": "public-read"})
        region = s3.meta.region_name
        return f"https://{bucket_name}.s3.{region}.amazonaws.com/{key}"
    except ClientError as e:
        raise RuntimeError(f"Failed to upload file to S3: {e}")
