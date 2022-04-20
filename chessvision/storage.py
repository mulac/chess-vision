""" Storage facade for files.  Currently only implemented for aws s3 """

import os
import logging

from . import util


class S3Storage():
    def __init__(self, dir, bucket="chess-vision"):
        if not os.path.exists(dir): os.mkdir(dir)
        self.__s3 = None
        self.dir = dir
        self.bucket = bucket
        logging.info(f'{self}: created.')

    def _s3(self):
        """ Avoids importing boto3 if s3 is never needed """
        if self.__s3 is None:
            import boto3
            self.__s3 = boto3.client('s3')
        return self.__s3

    def __call__(self, object):
        """ Fetches an object s3 if not already on disk """
        path = os.path.join(self.dir, object)
        logging.info(f'{self}: fetching {object}')
        if not os.path.exists(path): self.download(object, path)
        return path

    def __repr__(self):
        return f"S3Storage({self.dir} bucket={self.bucket})"

    def download(self, object, path):
        from botocore.exceptions import ClientError
        try: self._s3().download_file(self.bucket, object, path)
        except ClientError as e:
            logging.error(f'{self}: failed to download {object} to {path} [{e}]')

    def upload(self, object, path):
        from botocore.exceptions import ClientError
        logging.info(f'{self}: uploading {object} to {path}...')
        try: self._s3().upload_file(path, self.bucket, object)
        except ClientError as e:
            logging.error(f'{self}: failed to upload {object} from {path} [{e}]')
        

Storage = S3Storage(os.getenv(util.STORAGE_ENV) or "games")