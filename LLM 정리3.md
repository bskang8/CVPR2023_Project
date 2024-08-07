## 3. 검색증강생성모델 (RAG, LLM에 외부 지식을 제공)

- 산업 환경에서는 비용을 고려하고, 개인정보를 보호하며 신뢰할 수 솔루션이 가장 바람직합니다. 특히, 스타트업과 같은 기업들은 투자 대비 수익(Return on Investment, RoI)이 보이지 않는 상황에서 인재나 교육 모델에 처음부터 투자하려고 하지 않습니다.

- 최신 연구와 새로운 챗봇의 발표에서, 챗봇이 내부 파라미터(학습된 데이터)에 포함되지 않은 지식과 정보를 활용할 수 있는 능력을 가지고 있음이 밝혀졌다. 이 패러다임은 검색생성증강(RAG) 이라고 불린다.

- RAG는 비싼 미세 조정 없이 컨텍스트 학습을 가능하게 하여 대형 언어 모델(LLMs)의 사용을 더 비용 효율적으로 만든다. 기업은 RAG를 활용하여 새로운 데이터를 기반으로 응답을 처리하고 생성하는데 동일한 모델을 사용할 수 있고 솔루션을 맞춤화 하면서도 관련성을 유지할 수 있다. RAG는 또한 환각 문제를 완화하는 데 도움을 준다.

- 이를 달성할 수 있는 여러 가지 방법이 있으며, 첫 번째로는 필요한 정보를 추출하기 위해 다른 언어 모델을 반복적으로 호출하는 것이다.

- 아래 이미지에서 (출처) 반복적으로 언어 모델을 호출하는 방법을 살펴볼 수 있다:

- Another method for LLM gaining external knowledge is through information retrieval via memory units such as an external database, say of recent facts. As such, there are two types of information retrievers, dense and sparse.
    - As the name suggests, sparse retrievers use sparse bag of words representation of documents and queries while dense (neural) retrievers use dense query and document vectors obtained from a neural network.

- Yet another method is to leverage using agents which utilize APIs/tools to carry out a specializes task. The model chooses the most appropriate tool corresponding to a given input. With the help of tools like Google Search, Wikipedia and OpenAPI, LLMs can not only browse the web while responding, but also perform tasks like flight booking and weather reporting. LangChain offers a variety of different tools.

- “Even though the idea of retrieving documents to perform question answering is not new, retrieval-augmented LMs have recently demonstrated strong performance in other knowledge-intensive tasks besides Q&A. These proposals close the performance gap compared to larger LMs that use significantly more parameters.” (source)

- “With RAG, the external data used to augment your prompts can come from multiple data sources, such as a document repositories, databases, or APIs. The first step is to convert your documents and any user queries into a compatible format to perform relevancy search.

- To make the formats compatible, a document collection, or knowledge library, and user-submitted queries are converted to numerical representations using embedding language models. Embedding is the process by which text is given numerical representation in a vector space.

- RAG model architectures compare the embeddings of user queries within the vector of the knowledge library. The original user prompt is then appended with relevant context from similar documents within the knowledge library. This augmented prompt is then sent to the foundation model. You can update knowledge libraries and their relevant embeddings asynchronously.” source




