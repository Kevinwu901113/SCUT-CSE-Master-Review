# 论文目录

- #### GraphRAG综述：Graph Retrieval-Augmented Generation: A Survey  

- #### 微软GraphRAG：From Local to Global: A Graph RAG Approach to Query-Focused Summarization  

- #### KGRAG：KG-RAG: Bridging the Gap Between Knowledge and Creativity  

- #### communityKG：CommunityKG-RAG: Leveraging Community Structures in Knowledge Graphs for Advanced Retrieval-Augmented Generation in Fact-Checking  

- #### 开放领域问答学习框架：Retrieve What You Need: A Mutual Learning Framework for Open-domain Question Answering  

  - 两阶段学习框架
    1. 知识选择器：透过上下文选择与问题最相关的知识
    2. 基于FiD的阅读器：能够进行跨文段推理，从不同文段总结关键信息后融合再输出。（FiD是一种transformer结构的seq2seq模型）

- #### RAGforNLP：Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks  

  - RAG基础论文

- #### REAR：一种用于开放域问答的相关性感知检索增强框架：REAR:A Relevance-Aware Retrieval-Augmented Framework for
Open-Domain Question Answering

  - 设计了一个相关感知检索增强框架
    1. 相关性评估：生成相关性嵌入后进行打分。用来评估文档和查询的相关性
    2. 给予相关性引导生成：生成引导向量，确保平衡内部知识和外部文档知识
    3. 最终答案路由
       1. 透过路劲可靠性路由选择相关性评分最高的答案，简单、高效、快速
       2. 透过知识一致性路由选择一致性较高的答案，结合内部知识外部知识输出具有高一致性的答案

- #### 用于开放域问答的生成器-检索器-生成器方法：Generator-Retriever-Generator Approach for Open-Domain Question Answering

  - 








# 注

数据集：NaturalQuestions (NQ)、TriviaQA (TQA)和WebQ



