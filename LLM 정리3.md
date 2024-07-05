## 3. 검색증강생성모델 (RAG, LLM에 외부 지식을 제공)

- 산업 환경에서는 비용을 고려하고, 개인정보를 보호하며 신뢰할 수 솔루션이 가장 바람직합니다. 특히, 스타트업과 같은 기업들은 투자 대비 수익(Return on Investment, RoI)이 보이지 않는 상황에서 인재나 교육 모델에 처음부터 투자하려고 하지 않습니다.

- In most recent research and release of new chatbots, it’s been shown that they are capable of leveraging knowledge and information that is not necessarily in its weights. This paradigm is referred to as Retrieval Augmented Generation (RAG).

- RAG enables in-context learning without costly fine-tuning, making the use of LLMs more cost-efficient. By leveraging RAG, companies can use the same model to process and generate responses based on new data, while being able to customize their solution and maintain relevance. RAG also helps alleviate hallucination.

- There are several ways we can accomplish this, first of those being leveraging another LM by iteratively calling it to extract information needed.

- In the image below (source), we get a glimpse into how iteratively calling LM works:

- Another method for LLM gaining external knowledge is through information retrieval via memory units such as an external database, say of recent facts. As such, there are two types of information retrievers, dense and sparse.
    - As the name suggests, sparse retrievers use sparse bag of words representation of documents and queries while dense (neural) retrievers use dense query and document vectors obtained from a neural network.

- Yet another method is to leverage using agents which utilize APIs/tools to carry out a specializes task. The model chooses the most appropriate tool corresponding to a given input. With the help of tools like Google Search, Wikipedia and OpenAPI, LLMs can not only browse the web while responding, but also perform tasks like flight booking and weather reporting. LangChain offers a variety of different tools.

- “Even though the idea of retrieving documents to perform question answering is not new, retrieval-augmented LMs have recently demonstrated strong performance in other knowledge-intensive tasks besides Q&A. These proposals close the performance gap compared to larger LMs that use significantly more parameters.” (source)

- “With RAG, the external data used to augment your prompts can come from multiple data sources, such as a document repositories, databases, or APIs. The first step is to convert your documents and any user queries into a compatible format to perform relevancy search.

- To make the formats compatible, a document collection, or knowledge library, and user-submitted queries are converted to numerical representations using embedding language models. Embedding is the process by which text is given numerical representation in a vector space.

- RAG model architectures compare the embeddings of user queries within the vector of the knowledge library. The original user prompt is then appended with relevant context from similar documents within the knowledge library. This augmented prompt is then sent to the foundation model. You can update knowledge libraries and their relevant embeddings asynchronously.” source




