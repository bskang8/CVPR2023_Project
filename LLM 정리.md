## 0.개요
- LLM(Large Language Model)은 Transformer 아키텍처를 활용하는 심층 신경망입니다. LLM은 엄청난 양의 비정형 데이터를 활용하여 비지도 학습을 하였기 때문에 foundation model class의 일부로 알려져 있으며 fine-tuning과 같은 downstream task를 통해 특정 역할을 수행하는 다양한 모델로 변형될 수 있습니다.

- Transformer 아키텍처는 encoders와 decoders라는 두 부분으로 구성됩니다. Encoders와 decoders는 몇 가지 차이점을 제외하고는 거의 동일한 구조를 가지고 있습니다. (이에 대한 자세한 내용은 [Transformer](https://aman.ai/primers/ai/transformers/#transformer-encoder-and-decoder) 아키텍처 입문서에서 확인하세요.) 또한 인코더와 디코더 스택의 장단점은 [Autoregressive vs. Autoencoder Models](https://aman.ai/primers/ai/autoregressive-vs-autoencoder-models/)를 참조하세요.

- 생성적 인공지능 분야에서 디코더 기반 모델이 널리 사용되고 있음을 감안할 때, 이 글에서는 encoder models (예: BERT 및 그 변형)보다는 decoder models (예: GPT-x)에 더 중점을 두려고 합니다. 이후로는 LLM이라는 용어는 decoder 기반 모델로 사용됩니다.

- 주어진 텍스트 "prompt"가 주어졌을 때, 이 시스템이 본질적으로 하는 일은 시스템이 알고 있는 모든 "vocabulary" (단어들의 목록 - 단어의 부분 또는 토큰)에 대한 확률 분포를 계산하는 것입니다. Vocabulary는 사람 설계자가 시스템에 부여합니다. 예를 들어 GPT-3에는 약 50,000개 토큰의 vocabulary가 있습니다. [Source](https://aiguide.substack.com/p/on-detecting-whether-text-was-generated)

- LLM은 여전히 hallucination이나 chain of thought (최근 개선이 있음) 같은 수많은 제약 사항을 지니고 있지만, 해당 모델은 통계적 언어 모델링을 수행하도록 학습되었다는 점을 명심하는 것이 중요합니다.

## 1.Embeddings
