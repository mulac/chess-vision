""" Storage facade for files.  Currently only implemented for aws s3 """

import os
import logging


class S3Storage():
    def __init__(self, dir="games", bucket="chess-vision"):
        if not os.path.exists(dir): os.mkdir(dir)
        self.__s3 = None
        self.dir = dir
        self.bucket = bucket

    def _s3(self):
        """ Avoids importing boto3 if s3 is never needed """
        if self.__s3 is None:
            import boto3
            self.__s3 = boto3.client('s3')
        return self.__s3

    def __call__(self, object):
        """ Fetches an object s3 if not already on disk """
        if not os.path.exists(path := os.path.join(self.dir, object)):
            from botocore.exceptions import ClientError
            try:
                self._s3().download_file(self.bucket, object, path)
            except ClientError as e:
                logging.error(f'failed to download {object} from {self.bucket} to {path}')
        return path

    def upload(self, object):
        self._s3().upload_file(os.path.join(self.dir, object), self.bucket, object)


Storage = S3Storage(dir="/tmp/chess-vision")