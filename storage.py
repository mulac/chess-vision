import os
import boto3

from botocore.exceptions import ClientError

class StorageException(RuntimeError):
    pass

_s3 = boto3.client('s3')

def _upload(object, bucket, path):
    _s3.upload_file(path, bucket, object)

def _download(object, bucket, path):
    if not os.path.exists(path):
        try:
            _s3.download_file(bucket, object, path)
        except ClientError as e:
            print(f'FAILED to download {object} from {bucket} to {path}')
    return path


def Storage(object, dir="games", bucket="chess-vision", upload=False):
    if upload:
        return _upload(object, bucket, os.path.join(dir, object))
    return _download(object, bucket, os.path.join(dir, object))