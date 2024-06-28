## 0. 개요

- LLM(Large Language Model)은 트랜스포머(transformer) 아키텍처를 활용하는 심층 신경망입니다. LLM은 엄청난 양의 비정형 데이터를 비지도 학습한 foundation model의 한 종류이며 fine-tuning을 통해 다향한 종류의 downstream task 모델로 변형될 수 있습니다.

- Transformer 구조는 크게 encoders와 decoders로 구성됩니다. Encoders와 decoders는 네트워크의 구조적인 측면으로 바라보면 몇 가지 차이점을 제외하고는 거의 동일한 구조를 가지고 있습니다. (자세한 내용은 [Transformer](https://aman.ai/primers/ai/transformers/#transformer-encoder-and-decoder) 입문과 [Autoregressive vs. Autoencoder Models](https://aman.ai/primers/ai/autoregressive-vs-autoencoder-models/)를 참조하세요)
  
- 아울러, 생성형 인공지능은 디코더 기반 모델이 주로 사용되고 있기 때문에, 본 글에서는 encoder models (예: BERT 및 그 변형)보다는 decoder models (예: GPT-x)에 더 중점을 두려고 합니다. 이후 LLM이라는 용어는 decoder기반 모델을 지칭하고자 합니다.

- 주어진 텍스트 "prompt"가 주어졌을 때, 이 시스템이 본질적으로 하는 일은 시스템이 알고 있는 모든 "vocabulary" (단어들의 목록 - 단어의 부분 또는 토큰)에 대한 확률 분포를 계산하는 것입니다. Vocabulary는 사람이 설계하여 시스템에 부여합니다. 따라서 vocabulary는 시스템마다 다를 수 있으며 GPT-3의 경우 약 50,000개 토큰의 vocabulary가 있습니다. ([Source](https://aiguide.substack.com/p/on-detecting-whether-text-was-generated))

- LLM은 여전히 환각현상(hallucination)이나 chain of thought(최근 개선이 있음)같은 수많은 제약 사항을 지니고 있지만, 해당 모델은 통계적 언어 모델링을 수행하도록 학습되었다는 점을 명심하는 것이 중요합니다.

## 1. Embeddings

- 자연어 처리(NLP)에서의 임베딩은 단어 또는 문장의 의미론적 및 구문론적 속성을 포착하는 단어나 문장의 밀집된 벡터 표현입니다. 이러한 임베딩은 일반적으로 대규모 텍스트 모음을 BERT 및 그 변형, Word2Vec, Glove 또는 FastText와 같은 모델의 학습을 통해 얻을 수 있으며, 텍스트 정보를 기계 학습 알고리즘이 처리할 수 있는 형식으로 변환하는 방법을 제공합니다. 간단히 말해서, 임베딩은 단어의 의미론적 의미(내부적으로 하나 이상의 토큰으로 표시됨) 또는 문장의 의미론적 및 구문론적 속성을 조밀한 저차원 벡터로 표현하여 캡슐화합니다.

- 임베딩은 contextualized와 non-contextualized로 구분된다. 여기서 contextualized에서 각 토큰들의 임베딩은 input 주변의 다른 토큰들의 함수로 나타내어 진다. 그래서 “bank”와 같은 다의어 단어는 해당 단어가 “finance” 또는 “river” 컨텍스트에서 발생하는지 여부에 따라 고유한 임베딩을 가질 수 있습니다. 반면에 non-contextualized에서 각 토큰들의 임베딩은 컨택스트와 관계없이 사전학습을 통해 정적으로 얻어지며 downstream 작업에 활용될 수 있습니다.

- 토큰에 대한 임베딩을 얻으려면 각 단어에 대해 훈련된 모델에서 학습된 가중치를 추출합니다. 이러한 가중치는 단어 임베딩을 형성하며, 해당 임베딩은 각 단어의 조밀한 벡터로 표현됩니다.

### 1.1. Contextualized vs. Non-Contextualized Embeddings

- Transformer기반은 BERT (Bidirectional Encoder Representations from Transformers)와 같은 인코더 모델들은 contextualized embeddings를 생성하도록 설계 되었습니다. 각 단어에 적정한 벡터를 할당하는 기존의 단어 임베딩(Word2Vec 또는 GloVe)과는 달리 이러한 모델들은 단어의 문맥(주변 단어들)을 고려합니다. 문맥 안에서 단어들이 어떻게 사용되는지에 따라 동일한 단어도 다른 뜻을 지니기 때문에 이러한 모델은 단어에 대한 더 풍부하고 미묘한 의미를 포착할 수 있습니다.

### 1.2. Use-cases of Embeddings

- 임베딩을 통해 특정작업 수행을 위한 다양한 산술연산을 할수 있습니다 :
  1. **Word similarity** : 두 단어의 임베딩을 비교하여 유사성을 이해할 수 있습니다. 유사성 비교를 위해 코사인 유사도를 주로 사용합니다. 이는 두 벡터사이를 이루는 각도의 코사인 값을 측정하는 방법입니다. 두 벡터 사이에 코사인 값이 높다는 것은 두 단어들의 사용법이나 의미적인 측면에서 유사도가 높다는 것은 나타냅니다.
  2. **Word analogy** : 벡터연산은 단어 유추작업에도 사용할 수 있습니다. 예를 들어 "남자"와 "여자"가 주어지고 이와 유사한 기준으로 왕은 무엇과 대응하는 지를 유추하는 문제가 주어졌을 때 "왕" - "남자" + "여자"의 산술연산을 각 단어에 대응되는 임베딩벡터의 연산을 통해서 "여왕" 이라는 답을 얻을 수 있습니다.
  3. **Sentence similarity** : 두 문장 간의 유사성을 측정하려면 문장의 총 의미를 캡처하도록 설계된 BERT와 같은 모델에서 생성된 특수 [CLS] 토큰 임베딩을 사용할 수 있습니다. 또는 각 문장에 있는 모든 토큰의 임베딩을 평균화하는 평균 벡터를 만들어 해당 벡터들을 비교할 수 있습니다. 하지만 문장 유사성과 같은 문장 수준 작업의 경우 BERT 모델을 수정한 Sentence-BERT(SBERT)가 더 나은 선택인 경우가 많습니다. SBERT는 의미 공간에서 직접적으로 비교할 수 있는 문장 임베딩을 생성하도록 특별히 훈련되었으며, 이는 일반적으로 문장 수준 작업에서 더 나은 성능을 제공합니다. SBERT에서는 두 문장이 동시에 모델에 입력되므로 각 문장의 맥락을 다른 문장과 관련하여 이해할 수 있으므로 더 정확한 문장 임베딩이 가능합니다.

### 1.3. Similarity Search with Embeddings
- 인코더 모델의 출력으로 contextualized embedding을 얻게 됩니다. 두 단어간의 유사성 이해, 단어 유추등과 같은 다양한 작업을 위해 임베딩에 대한 산술연산을 할수 있습니다.

- Word simility 작업에서는 단어들에 대한 각각의 contextualized embedding을 사용할 수 있습니다. 반면에 sentence similarity 작업에서는 [CLS] 토근의 output을 사용할 수 있고 또한 모든 단어 토큰들의 임베딩을 평균화한 임베딩벡터를 사용할 수 있습니다. 하지만 sentence similarity 작업에서 최상의 성능을 얻으려면 Sentence BERT 또는 그 변형 모델들이 선호됩니다.

- Word/sentence similarity는 두 단어/문장의 의미가 의미적으로 동일한 정도를 측정한 것입니다.

- 다음은 word/sentence similarity에 대한 가장 일반적인 두 가지 척도입니다(두 가지 모두 "거리 척도"는 아닙니다.)

#### 1.3.1. Dot Product Similarity
- 두 벡터 $u$와 $v$의 dot product는 다음과 같이 정의된다 :
  
  $$
  u \cdot v = |u||v| cos\theta
  $$
