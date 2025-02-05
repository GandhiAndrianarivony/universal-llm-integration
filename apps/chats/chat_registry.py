import streamlit as st

from core.factory import Factory
from .chat_interfaces import (
    XlsxChat,
    WebsearchChat,
    BaseChatUI,
)


chat_factory = Factory()
chat_factory.register(name=XlsxChat.chat_type, creator=XlsxChat)
chat_factory.register(name=WebsearchChat.chat_type, creator=WebsearchChat)


# @st.cache_resource
def create_chat_interface(chat_interface: str) -> BaseChatUI:
    return chat_factory.create(chat_interface)
