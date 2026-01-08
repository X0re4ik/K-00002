from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv

from ._get_device import get_device

load_dotenv()


class BaseConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
    )


class APPConfig(BaseConfig):
    model_config = SettingsConfigDict(env_prefix="APP_")

    develop: bool = Field(default=False)
    logger_level: str = Field(default="INFO")

    device: str = Field(
        default_factory=get_device,
    )


class QdrantConfig(BaseConfig):
    model_config = SettingsConfigDict(env_prefix="QDRANT_")

    host: str = Field(
        default="localhost",
    )
    use_https: bool = Field(
        default=False,
    )
    port: int = Field(
        default=6333,
    )

    text_collection: str = Field(
        default="text",
    )

    @property
    def http_url(self):
        http_ = "https" if self.use_https else "http"
        return f"{http_}://{self.host}:{self.port}"


class APIConfig(BaseConfig):

    qdrant: QdrantConfig = Field(
        default_factory=QdrantConfig,
    )


class Settings(BaseConfig):
    app: APPConfig = Field(default_factory=APPConfig)

    api: APIConfig = Field(
        default_factory=APIConfig,
    )
