# Ollama+GraphRAG(llama3.1)

使用ollma的llama3.1结合GraphRAG的尝试

### 安装Ollama

`conda create -n ollama python=3.10     #推荐python版本使用3.10`

`conda activate ollama`

`pip install ollama`

- 使用Ollama下载模型
  - 需要一个LLM和一个Embedding模型

这里使用llama3.1(默认8B)和nomic-embed-text

`ollama pull llama3.1     #llm`

`ollama pull nomic-embed-text`

`ollama list      #查看是否下载成功，应该显示llama3.1:latest和nomic-embed-text`

`ollama serve     #启动服务，默认使用11434端口`

### 下载GraphRAG源码

- **源码需要用git下载，不能github官网直接download，会报错**

`git clone https://github.com/TheAiSingularity/graphrag-local-ollama.git`

`cd graphrag-local-ollama     #进入目录`

### 安装依赖

`pip install -e .     #如果前面直接在github网页download在这一步会报错`

### 创建工作目录

`mkdir -p ./ragtest/input         #这一步应该已经处于graphrag-local-ollama文件夹内`

- 将你的input放到./ragtest/input内

### 项目初始化

`python -m graphrag.index --init --root ./ragtest`

- 初始化完成后应该包含以下
  - output
  - input
  - settings.yaml
  - prompts
  - .env(隐藏文件)

#### （可选）使用默认配置文件

`mv settings.yaml ./ragtest`

### 修改配置文件

- llm内的model需修改成你在Ollama list指令时看到的llama名称，例如：llama3.1:latest
- embeddings内的model需修改成Ollma list指令时看到的embedding名称，例如：nomic-embed-text

### 构建Graph，进行索引

`python -m graphrag.index --root ./ragtest`

构建完成后可以进行查询（此处为全局查询）

`python -m graphrag.query --root ./ragtest --method global "$你的问题"`

至此，Graph结合Ollama完成

# 额外补充

### 局部查询

- 原作并未完善局部查询，若要进行局部查询，需要对源码进行修改

`pip install langchain_community        #安装依赖项`

- 修改文件

`file://graphrag-local-ollama/graphrag/query/llm/oai/embedding.py`

修改为以下内容

```python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License
"""OpenAI Embedding model implementation."""

import asyncio
from collections.abc import Callable
from typing import Any

import numpy as np
import tiktoken
from tenacity import (
    AsyncRetrying,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from graphrag.query.llm.base import BaseTextEmbedding
from graphrag.query.llm.oai.base import OpenAILLMImpl
from graphrag.query.llm.oai.typing import (
    OPENAI_RETRY_ERROR_TYPES,
    OpenaiApiType,
)
from graphrag.query.llm.text_utils import chunk_text
from graphrag.query.progress import StatusReporter

from langchain_community.embeddings import OllamaEmbeddings

class OpenAIEmbedding(BaseTextEmbedding, OpenAILLMImpl):
    """Wrapper for OpenAI Embedding models."""

    def __init__(
        self,
        api_key: str | None = None,
        azure_ad_token_provider: Callable | None = None,
        model: str = "nomic-embed-text",
        deployment_name: str | None = None,
        api_base: str | None = None,
        api_version: str | None = None,
        api_type: OpenaiApiType = OpenaiApiType.OpenAI,
        organization: str | None = None,
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        max_retries: int = 10,
        request_timeout: float = 180.0,
        retry_error_types: tuple[type[BaseException]] = OPENAI_RETRY_ERROR_TYPES,  # type: ignore
        reporter: StatusReporter | None = None,
    ):
        OpenAILLMImpl.__init__(
            self=self,
            api_key=api_key,
            azure_ad_token_provider=azure_ad_token_provider,
            deployment_name=deployment_name,
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,  # type: ignore
            organization=organization,
            max_retries=max_retries,
            request_timeout=request_timeout,
            reporter=reporter,
        )

        self.model = model
        self.encoding_name = encoding_name
        self.max_tokens = max_tokens
        self.token_encoder = tiktoken.get_encoding(self.encoding_name)
        self.retry_error_types = retry_error_types

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's sync function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        Please refer to: https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        for chunk in token_chunks:
            try:
                embedding, chunk_len = self._embed_with_retry(chunk, **kwargs)
                chunk_embeddings.append(embedding)
                chunk_lens.append(chunk_len)
            except Exception as e:  # noqa BLE001
                self._reporter.error(
                    message="Error embedding chunk",
                    details={self.__class__.__name__: str(e)},
                )
                continue
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    async def aembed(self, text: str, **kwargs: Any) -> list[float]:
        """
        Embed text using OpenAI Embedding's async function.

        For text longer than max_tokens, chunk texts into max_tokens, embed each chunk, then combine using weighted average.
        """
        token_chunks = chunk_text(
            text=text, token_encoder=self.token_encoder, max_tokens=self.max_tokens
        )
        chunk_embeddings = []
        chunk_lens = []
        embedding_results = await asyncio.gather(*[
            self._aembed_with_retry(chunk, **kwargs) for chunk in token_chunks
        ])
        embedding_results = [result for result in embedding_results if result[0]]
        chunk_embeddings = [result[0] for result in embedding_results]
        chunk_lens = [result[1] for result in embedding_results]
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)  # type: ignore
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)
        return chunk_embeddings.tolist()

    def _embed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = Retrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            for attempt in retryer:
                with attempt:
                    embedding = (
                        OllamaEmbeddings(
                            model=self.model,
                        ).embed_query(text)
                        or []
                    )
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)

    async def _aembed_with_retry(
        self, text: str | tuple, **kwargs: Any
    ) -> tuple[list[float], int]:
        try:
            retryer = AsyncRetrying(
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential_jitter(max=10),
                reraise=True,
                retry=retry_if_exception_type(self.retry_error_types),
            )
            async for attempt in retryer:
                with attempt:
                    embedding = (
                        await OllamaEmbeddings(
                            model=self.model,
                        ).embed_query(text) or [] )
                    return (embedding, len(text))
        except RetryError as e:
            self._reporter.error(
                message="Error at embed_with_retry()",
                details={self.__class__.__name__: str(e)},
            )
            return ([], 0)
        else:
            # TODO: why not just throw in this case?
            return ([], 0)

```





### 全局搜索时出现json类似报错，是因为llm的输出格式不固定，以及llm忽略system身份的prompt（未知原因）

#### 修改全局搜索search文件

File: /graphrag/query/structured_search/global_search/search.py , method: _map_response_single_batch

注释177~180，并添加下列

`search_messages = [ {"role": "user", "content": search_prompt + "\n\n### USER QUESTION ### \n\n" + query} ]`