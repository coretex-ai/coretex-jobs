from typing import Optional
from http import HTTPStatus

import logging

from coretex import currentTaskRun, Model, AWSSecret

import boto3
import botocore.exceptions


def main() -> None:
    taskRun = currentTaskRun()

    model: Optional[Model] = taskRun.parameters["model"]
    if model is None:
        logging.warning(">> [S3 Upload] Model not provided, skipping upload")
        return

    model.download()

    secretName = taskRun.parameters["secret"]
    secret = AWSSecret.fetchByName(secretName)

    resource = boto3.resource(
        service_name = "s3",
        aws_access_key_id = secret.key,
        aws_secret_access_key = secret.value
    )

    bucketName = taskRun.parameters["bucket"]

    # Check if the bucket exists
    try:
        resource.meta.client.head_bucket(Bucket = bucketName)
    except botocore.exceptions.ClientError as exception:
        errorCode = int(exception.response["Error"]["Code"])

        if errorCode == HTTPStatus.NOT_FOUND:
            raise ValueError(f">> [S3 Upload] Bucket \"{bucketName}\" not found")

        if errorCode == HTTPStatus.FORBIDDEN:
            raise ValueError(f">> [S3 Upload] Bucket \"{bucketName}\" access denied")

        errorMessage = exception.response["Error"]["Message"]
        raise ValueError(f">> [S3 Upload] Failed to fetch bucket \"{bucketName}\". Reason: {errorMessage}")

    destination = taskRun.parameters["destination"]
    if destination is None:
        destination = f"{model.name}.zip"

    logging.info(f">> [S3 Upload] Uploading model to S3 bucket \"{bucketName}\", destination \"{destination}\"")

    bucket = resource.Bucket(bucketName)

    if any(obj.key == destination for obj in bucket.objects.all()):
        logging.warning(f">> [S3 Upload] Destination \"{destination}\" already exists, this operation will overwrite it")

    bucket.upload_file(model.zipPath, destination)


if __name__ == "__main__":
    main()
