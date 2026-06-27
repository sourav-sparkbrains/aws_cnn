from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):

    #Data settings
    RAW_DATA_DIR: ClassVar = BASE_DIR / 'data' / 'raw'
    PROCESSED_DATA_DIR: ClassVar = BASE_DIR / 'data' / 'processed'
    IMAGE_SIZE:int = Field(default=224)
    BATCH_SIZE: int = Field(default=32)

    #Training settings
    EPOCHS: int = Field(default=10)
    LEARNING_RATE: float = Field(default=0.001)
    NO_OF_CLASSES: int = Field(default=3)
    TRAIN_SIZE: float = Field(default=0.8)

    #AWS settings
    AWS_ACCESS_KEY: str = Field(...)
    AWS_SECRET_KEY: str = Field(...)

    #S3 settings
    S3_BUCKET_NAME: str = Field(...)

    #SAGEMAKER settings
    SAGEMAKER_ARN_ROLE: str = Field(...)

    #MLFLow settings
    EXPERIMENT_NAME: str = Field(...)
    RUN_NAME: str = Field(...)

    model_config = SettingsConfigDict(
        env_file='.env',
        extra='ignore',
        case_sensitive=False
    )

settings = Settings()