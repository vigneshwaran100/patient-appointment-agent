from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration and environment settings."""

    # LLM Settings (Supports Gemini & Groq)
    llm_provider: str = Field(default="auto", alias="LLM_PROVIDER")  # "auto", "gemini", "groq"
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")
    temperature: float = Field(default=0.0, alias="TEMPERATURE")

    # Storage Settings
    database_path: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parent.parent / "data" / "clinic.db"),
        alias="DATABASE_PATH",
    )

    # Agent / Policy Settings
    policy_store_path: str = Field(
        default_factory=lambda: str(
            Path(__file__).resolve().parent.parent / "improvement" / "policies"
        ),
        alias="POLICY_STORE_PATH",
    )
    max_clarification_turns: int = Field(default=3, alias="MAX_CLARIFICATION_TURNS")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()
