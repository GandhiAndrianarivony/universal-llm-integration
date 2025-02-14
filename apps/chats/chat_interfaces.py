import os
import tempfile
from abc import ABC, abstractmethod
import pandas as pd

import streamlit as st

from llama_index.core import SimpleDirectoryReader
from llm.query_engines import (
    XLSXQueryEngine,
    WebsearchQueryEngine,
    IQueryEngine,
    ArxivQueryEngine,
)


class IChatUI(ABC):
    @abstractmethod
    def content(self):
        pass

    @property
    def query_engine(self) -> IQueryEngine:
        pass


class XlsxChat(IChatUI):
    chat_type = "XLSX"
    _prompt = """
            Context information is below.\n
            ---------------------\n
            {context_str}\n
            ---------------------\n
            You are an assistant for question-answering task.
            Use the above context to answer the question.
            I want you to think step by step to answer the query in a highly precise and crisp manner focused on the final answer
            If you don't know the answer, just say that you don't know.
            Don't Generate any programming language code.\n

            Query: {query_str}\n
            Answer: 
        """

    def __init__(self):
        self._docs = None

    def content(self):
        st.header("Ajouter vos documents!")

        uploaded_file = st.file_uploader(
            "Choose your `.xlsx` file",
            type=["xlsx"],
            key="xlsx chat",
        )
        if uploaded_file:
            with tempfile.TemporaryDirectory(dir="./data") as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if os.path.exists(temp_dir):
                    loader = SimpleDirectoryReader(
                        input_files=[file_path],
                    )
                    self._docs = loader.load_data()
                else:
                    st.error(
                        "Could not find the file you uploaded, please check again..."
                    )
                    st.stop()

                XlsxChat._display_excel(uploaded_file)
                st.success("Ready to Chat!")

    @property
    def query_engine(self):
        if self._docs is not None:
            return XLSXQueryEngine(
                docs=self._docs,
                prompt=XlsxChat._prompt,
            )

        st.warning("Upload an xlsx file")

    @staticmethod
    def _display_excel(file):
        st.markdown("### Excel Preview")
        # Read the Excel file
        df = pd.read_excel(file)
        # Display the dataframe
        st.dataframe(df)


class WebsearchChat(IChatUI):
    chat_type = "Websearch"
    _prompt = """
        You are an assistant for web search task.
    """

    def __init__(self):
        super().__init__()

    def content(self):
        pass

    @property
    def query_engine(self):
        return WebsearchQueryEngine(prompt=WebsearchChat._prompt)


class ArxivChat(IChatUI):
    chat_type = "Arxiv"
    _prompt = """
        You are an AI assistant specialized in retrieving research articles from arXiv.  
        Your task is to search for the most relevant articles based on a given user query and provide the following details for each result:  

        - **Title**  
        - **Authors**  
        - **Publication Date**  
        - **Abstract**  
        - **PDF Link** (if available)  
        - **Category** (e.g., Artificial Intelligence, Physics, Mathematics)  

        Ensure that the results are **accurate, up-to-date, and sorted by relevance**.  
        If no relevant articles are found, respond with: **'I DIDN'T FIND ANY RELEVANT ARTICLE'**.  

        Maintain clarity and conciseness in the output while preserving essential details.  
    """




    def __init__(self):
        super().__init__()

    def content(self):
        pass

    @property
    def query_engine(self):
        return ArxivQueryEngine(prompt=ArxivChat._prompt)


chats = [XlsxChat.chat_type, WebsearchChat.chat_type, ArxivChat.chat_type]
