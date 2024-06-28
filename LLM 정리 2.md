## 2. LLM은 어떻게 동작하는 가

- 개요에서 언급한 것 처럼 LLM은 이전 토큰 세트를 기반으로 다음 토큰을 예측하도록 훈련되었습니다. 이는 생성기능을 활성화 하는 자동회귀 방식(autoregressive, 현재 생성된 토큰은 다음 토큰을 생성하기 위한 입력으로 거대언어모델에 재입력 됨)으로 수행하여 생성을 가능하게 합니다.

- The first step involves taking the prompt they receive, tokenining it, and converting it into embeddings, which are vector representations of the input text. Note that these embeddings are initialized randomly and learned during the course of model training, and represent a non-contextualized vector form of the input token.

첫 번째 단계에서 프롬프트를 입력 받으면 토큰화를 수행하고 임베딩으로 변환하는 작업(벡터화 작업)이 수행 됩니다. 이러한 임베딩은 모델 훈련 과정에서 무작위로 초기화되고 학습되며 입력 토큰의 맥락화되지 않은 벡터 형식을 나타냅니다.

- Next, they do layer-by-layer attention and feed-forward computations, eventually assigning a number or logit to each word in its vocabulary (in the case of a decoder model like GPT-N, LLaMA, etc.) or outputs these features as contextualized embeddings (in the case of an encoder model like BERT and its variants such as RoBERTa, ELECTRA, etc.).

다음으로, 레이어별 주의 및 피드포워드 계산을 수행하여 결국 어휘의 각 단어에 숫자 또는 로짓을 할당하거나(GPT-N, LLaMA 등과 같은 디코더 모델의 경우) 이러한 기능을 출력합니다. 상황에 맞는 임베딩으로 사용됩니다(BERT와 같은 인코더 모델 및 RoBERTa, ELECTRA 등과 같은 변형 모델의 경우).

- Finally, in the case of decoder models, the next step is converting each (unnormalized) logit into a (normalized) probability distribution (via the Softmax function), determining which word shall come next in the text.

마지막으로 디코더 모델의 경우 다음 단계는 각 (정규화되지 않은) 로짓을 (정규화된) 확률 분포(Softmax 함수를 통해)로 변환하여 텍스트에서 다음에 올 단어를 결정하는 것입니다.

- Let’s break the steps down into finer detail:

### 2.1. LLM 학습 단계

### 2.2. Reasoning
