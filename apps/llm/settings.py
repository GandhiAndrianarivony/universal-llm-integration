from pydantic_settings import BaseSettings, SettingsConfigDict


class HuggingFaceConfig(BaseSettings):
    HUGGINGFACE_API_KEY: str

    model_config = SettingsConfigDict(env_file=".env")


hf_configs = HuggingFaceConfig()
