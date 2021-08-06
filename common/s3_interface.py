import logging
import os
import sys
import threading

import boto3
from botocore.exceptions import ClientError

BUCKET = 'facial-landmark-detection-thesis'


def download_file_from_s3(local_file_name, s3_object_key, s3_bucket=BUCKET):
    s3 = boto3.client('s3')
    meta_data = s3.head_object(Bucket=s3_bucket, Key=s3_object_key)
    total_length = int(meta_data.get('ContentLength', 0))
    downloaded = 0

    def progress(chunk):
        nonlocal downloaded
        downloaded += chunk
        done = int(50 * downloaded / total_length)
        sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
        sys.stdout.flush()

    print(f'Downloading {s3_object_key}')
    with open(local_file_name, 'wb') as f:
        s3.download_fileobj(s3_bucket, s3_object_key, f, Callback=progress)


#
# def download_file_from_s3(file_name, destination, bucket=BUCKET):
#     s3 = boto3.client('s3')
#     s3.download_file(bucket, destination, file_name,
#                      Callback=ProgressPercentage(file_name))
#     logging.info(f'Downloading from S3, bucket {bucket}\t file {file_name}\t destination:{destination}')
#     return True


def upload_file_to_s3(file_name, bucket=BUCKET, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    logging.info(f'Uploading to S3, bucket {bucket}\t file {file_name}\t destination:{object_name}')

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name,
                                         Callback=ProgressPercentage(file_name))

    except ClientError as e:
        logging.error(e)
        return False
    return True


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()


def main():
    download_file_from_s3('/home/itamar/thesis/outputs/DETR/020821_004317/checkpoint/rootkey.csv', bucket=BUCKET,
                          object_name=None)


if __name__ == "__main__":
    main()
