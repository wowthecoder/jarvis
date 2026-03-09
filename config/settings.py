from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="JARVIS_")
    # API keys
    google_api_key: str = ""
    tavily_api_key: str = ""
    huggingface_token: str = ""

    # Model names
    manager_model: str = "deepseek-r1:8b"
    text_model: str = "llama3.1:8b"
    multimodal_model: str = "gemini-2.0-flash"
    web_model: str = "gemini-2.0-flash"
    extractor_model: str = "gemini-2.0-flash"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # Execution limits
    max_agent_iterations: int = 15
    agent_timeout_seconds: int = 120

    # Paths
    data_cache_dir: str = "data/gaia"
    output_dir: str = "outputs"



settings = Settings()
