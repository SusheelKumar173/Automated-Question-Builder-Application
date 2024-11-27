import boto3
import json
import logging
from botocore.exceptions import ClientError
import config

s3_client = boto3.client('s3')
sns_client = boto3.client('sns')
dynamodb_client = boto3.resource('dynamodb')
bedrock_client = boto3.client('bedrock-runtime') 

logger = logging.getLogger()
logger.setLevel(config.LOG_LEVEL)

def lambda_handler(event, context):
    """
    Main Lambda handler function.
    """
    try:
        logger.info("Event received: %s", json.dumps(event))
        
        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']

            logger.info(f"Processing file: {object_key} from bucket: {bucket_name}")

            curriculum_content = read_s3_file(bucket_name, object_key)

            topics = preprocess_curriculum(curriculum_content)

            question_bank = generate_questions_bedrock(topics)

            save_to_s3(question_bank, config.OUTPUT_BUCKET, f"questions/{object_key.split('/')[-1].replace('.csv', '_questions.json')}")

            save_metadata_to_dynamodb(object_key, len(question_bank))

            send_notification(f"Question bank generation for {object_key} is complete.", config.SNS_TOPIC_ARN)

        return {"statusCode": 200, "body": "Question bank generated successfully!"}
    
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return {"statusCode": 500, "body": str(e)}

def read_s3_file(bucket, key):
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        logger.info(f"Successfully read file: {key}")
        return content
    except ClientError as e:
        logger.error(f"Error reading file from S3: {str(e)}")
        raise

def preprocess_curriculum(content):
    topics = []
    lines = content.split("\n")
    for line in lines:
        if line.strip():
            topics.append(line.strip())
    logger.info(f"Extracted topics: {topics}")
    return topics

def generate_questions_bedrock(topics):
    try:
        prompt = {
            "prompt": f"Generate a question bank based on these topics: {', '.join(topics)}",
            "max_tokens": 300,
            "temperature": 0.7,
        }
        response = bedrock_client.invoke_model(
            modelId=config.BEDROCK_MODEL_ID,
            body=json.dumps(prompt),
            contentType="application/json"
        )
        response_body = json.loads(response['body'].read())
        questions = response_body.get('completions', [])
        logger.info("Questions generated successfully.")
        return questions
    except ClientError as e:
        logger.error(f"Error generating questions: {str(e)}")
        raise

def save_to_s3(data, bucket, key):
    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data))
        logger.info(f"Successfully saved file to S3: {key}")
    except ClientError as e:
        logger.error(f"Error saving file to S3: {str(e)}")
        raise

def save_metadata_to_dynamodb(file_name, question_count):
    try:
        table = dynamodb_client.Table(config.DYNAMODB_TABLE)
        table.put_item(
            Item={
                "FileName": file_name,
                "QuestionCount": question_count,
                "Status": "Completed",
                "Timestamp": boto3.dynamodb.conditions.Attr('NOW()')
            }
        )
        logger.info(f"Metadata saved for file: {file_name}")
    except ClientError as e:
        logger.error(f"Error saving metadata to DynamoDB: {str(e)}")
        raise

def send_notification(message, topic_arn):
    try:
        sns_client.publish(TopicArn=topic_arn, Message=message)
        logger.info(f"Notification sent: {message}")
    except ClientError as e:
        logger.error(f"Error sending notification: {str(e)}")
        raise
