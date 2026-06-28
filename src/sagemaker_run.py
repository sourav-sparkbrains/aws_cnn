import sagemaker
from sagemaker.pytorch import PyTorch

from src.config import settings

def run_training_job():

    session = sagemaker.Session()
    role = settings.SAGEMAKER_ARN_ROLE

    estimator = PyTorch(
        entry_point="train.py",
        source_dir='src',
        role=role,
        framework_version='2.0.1',
        py_version="py310",
        instance_type='ml.m5.large',
        instance_count=1,
        output_path=f's3://{settings.S3_BUCKET_NAME}/model-output',
        hyperparameters={
            'epochs': settings.EPOCHS,
            'batch_size': settings.BATCH_SIZE,
            'learning_rate': settings.LEARNING_RATE,
        }
    )

    estimator.fit({
        'train': f's3://{settings.S3_BUCKET_NAME}/data',
    })

if __name__ == '__main__':
    run_training_job()