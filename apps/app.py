# import os
import uuid

import streamlit as st

from llama_index.core import Settings

from llm.llm_providers import providers
from llm.llm_registry import create_provider_chat_model

from chats.chat_interfaces import chats
from chats.chat_registry import create_chat_interface


# TODO: Add chat memory


class Chat:
    def __init__(self):
        self.logged_user = st.session_state.get("logged_user")
        self.__post_init__()

    def __post_init__(self):
        if f"{self.logged_user}_session_id" not in st.session_state:
            st.session_state[f"{self.logged_user}_session_id"] = str(uuid.uuid4())
            st.session_state[f"{self.logged_user}_query_engine"] = dict()

    def content(self):
        with st.sidebar:
            selected_provider = st.selectbox(label="**Provider**", options=providers)
            chat_model = create_provider_chat_model(name=selected_provider)

            selected_model_name = st.selectbox(
                label=f"**{selected_provider} Model**", options=chat_model.options
            )

            # set up llm and embed_model
            # TODO: In case of not LLamaIndex
            llm, embed_model = chat_model.load(selected_model_name)
            Settings.llm = llm
            Settings.embed_model = embed_model

            chat_interface = st.selectbox(
                "**Chat with**", options=chats, index=None, on_change=self.reset_chat
            )

            # TODO: Instantiate chat UI
            if chat_interface is not None:
                chat_interface = create_chat_interface(chat_interface=chat_interface)
                chat_interface.content()
                query_engine = chat_interface.query_engine

            else:
                st.stop()

            is_streamed = st.checkbox("Stream", disabled=True, value=False)

        # Initialize chat history
        if f"{self.logged_user}_messages" not in st.session_state:
            self.reset_chat()

        col1, col2 = st.columns([6, 1])
        with col1:
            st.header(chat_interface.chat_type)
        with col2:
            st.button("Clear ↺", on_click=self.reset_chat)

        self.display_chat_history_messages()

        if prompt := st.chat_input("What's up?"):
            with st.chat_message("user"):
                st.markdown(prompt)

                self.save_message(message={"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                # STREAM RESPONSE
                if is_streamed:
                    for chunk in query_engine.query(prompt, streaming=is_streamed):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                else:
                    full_response = query_engine.query(prompt)

                message_placeholder.markdown(full_response)

                self.save_message(
                    message={"role": "assistant", "content": full_response}
                )

    def display_chat_history_messages(self):
        # Display chat messages from history on app rerun
        for message in st.session_state.get(f"{self.logged_user}_messages"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def save_message(self, message: dict):
        st.session_state[f"{self.logged_user}_messages"].append(message)

    def reset_chat(self):
        st.session_state[f"{self.logged_user}_messages"] = []


if __name__ == "__main__":
    chat = Chat()
    chat.content()
