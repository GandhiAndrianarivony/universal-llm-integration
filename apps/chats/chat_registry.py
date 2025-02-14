import streamlit as st

from core.factory import Factory
from .chat_interfaces import (
    IChatUI,
    XlsxChat,
    WebsearchChat,
    ArxivChat
)


chat_factory = Factory()
chat_factory.register(name=XlsxChat.chat_type, creator=XlsxChat)
chat_factory.register(name=WebsearchChat.chat_type, creator=WebsearchChat)
chat_factory.register(name=ArxivChat.chat_type, creator=ArxivChat)


# @st.cache_resource
def create_chat_interface(chat_interface: str) -> IChatUI:
    return chat_factory.create(chat_interface)
