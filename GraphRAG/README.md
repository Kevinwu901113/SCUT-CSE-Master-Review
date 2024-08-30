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

- 若要进行局部查询，需要对源码进行修改

`pip install langchain_community        #安装依赖项`
