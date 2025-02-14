from abc import ABC, abstractmethod

from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent

from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.tools.arxiv import ArxivToolSpec


class IQueryEngine[T](ABC):
    @abstractmethod
    def query(self, query: str, streaming: bool = False) -> T:
        pass

    @staticmethod
    @abstractmethod
    def create_engine(*args, **kwargs):
        pass


class ArxivQueryEngine(IQueryEngine):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def query(self, query: str, streaming: bool = False):
        query_engine = ArxivQueryEngine.create_engine(context=self.prompt)
        response = query_engine.query(query)
        return response

    @staticmethod
    def create_engine(context: str):
        tools = ArxivToolSpec().to_tool_list()

        agent = ReActAgent.from_tools(
            tools,
            llm=Settings.llm,
            verbose=True,
            context=context,
        )
        return agent


class XLSXQueryEngine(IQueryEngine):
    def __init__(self, docs: list, prompt: str):
        self.docs = docs
        self.prompt = prompt

    def query(self, query: str, streaming: bool = False):
        query_engine = XLSXQueryEngine.create_engine(
            _docs=self.docs,
            qa_prompt_tmpl_str=self.prompt,
            streaming=streaming,
        )
        response = query_engine.query(query)
        return response  # .response

    @staticmethod
    def create_engine(_docs, qa_prompt_tmpl_str: str, streaming: bool = False):
        # CREATE INDEX
        node_parser = MarkdownNodeParser()
        index = VectorStoreIndex.from_documents(
            _docs,
            transformations=[node_parser],
            show_progress=True,
        )

        # CREATE QUERY ENGINE
        query_engine = index.as_query_engine(streaming=streaming)
        qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

        query_engine.update_prompts(
            {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
        )
        return query_engine


class WebsearchQueryEngine(IQueryEngine):
    def __init__(self, prompt: str):
        self.prompt = prompt

    def query(self, query: str, streaming: bool = False):
        query_engine = WebsearchQueryEngine.create_engine(context=self.prompt)

        response = query_engine.query(query)
        return response  # .response

    @staticmethod
    def create_engine(context: str):
        tools = DuckDuckGoSearchToolSpec().to_tool_list()

        agent = ReActAgent.from_tools(
            tools,
            llm=Settings.llm,
            verbose=True,
            context=context,
        )
        return agent
