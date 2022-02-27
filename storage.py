import os
import boto3

_s3 = boto3.client('s3')

def _upload(object, bucket, path):
    _s3.upload_file(path, bucket, object)

def _download(object, bucket, path):
    if not os.path.exists(path):
        _s3.download_file(bucket, object, path)
    return path


def Storage(object, dir="games", bucket="chess-vision", upload=False):
    path = os.path.join(dir, object)
    if upload:
        return _upload(object, bucket, path)
    return _download(object, bucket, path)