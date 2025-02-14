from core.factory import Factory
from .llm_providers import IChatModel, HuggingFaceChatModel, OllamaChatModel


llm_factory = Factory()
llm_factory.register(
    name=HuggingFaceChatModel.provider_name,
    creator=HuggingFaceChatModel,
)
llm_factory.register(name=OllamaChatModel.provider_name, creator=OllamaChatModel)


# @st.cache_resource
def create_provider_chat_model(name: str) -> IChatModel:
    return llm_factory.create(name)
