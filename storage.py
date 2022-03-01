import os


_s3 = None


def Storage(object, dir="games", bucket="chess-vision", upload=False):
    if upload:
        return _upload(object, bucket, os.path.join(dir, object))
    return _download(object, bucket, os.path.join(dir, object))


def _get_s3():
    if _s3 is None:
        import boto3
        _s3 = boto3.client('s3')
    return _s3

def _upload(object, bucket, path):
    _get_s3().upload_file(path, bucket, object)


def _download(object, bucket, path):
    if not os.path.exists(path):
        from botocore.exceptions import ClientError
        try:
            _get_s3().download_file(bucket, object, path)
        except ClientError as e:
            print(f'FAILED to download {object} from {bucket} to {path}')
    return path
