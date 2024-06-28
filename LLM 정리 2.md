## 2. LLM은 어떻게 동작하는 가

- 개요에서 언급한 것 처럼 LLM은 이전 토큰 세트를 기반으로 다음 토큰을 예측하도록 훈련되었습니다. 이는 생성기능을 활성화 하는 자동회귀 방식(autoregressive, 현재 생성된 토큰은 다음 토큰을 생성하기 위한 입력으로 거대언어모델에 재입력 됨)을 수행하여 생성을 가능하게 합니다.

- 첫 번째 단계에서 받은 프롬프트를 토큰화 하고 이를 임베딩으로 변환하는 작업이 수행 됩니다. 임베딩은 입력 텍스트의 벡터 표현입니다. 이러한 임베딩은 무작위로 초기화되어 입력 토큰의 비 의미론적인 벡터 형태를 나타냅니다. 그리고, 모델 훈련 과정에서 맥락화되는 학습이 수행됩니다.

- 다음으로, 레이어별 어텐션(attentation) 및 피드포워드 연산을 수행하여 최종적으로 어휘의 각 단어에 숫자 또는 로짓(logit)을 출력하거나(GPT-N, LLaMA 등의 디코더 모델) 의미론적 임베딩을 출력합니다(BERT와 같은 인코더 모델 및 RoBERTa, ELECTRA 등과 같은 변형 모델).

- 마지막으로 디코더 모델의 경우 다음 단계는 각 (정규화되지 않은) 로짓을 (정규화된) 확률 분포(Softmax 함수를 통해)로 변환하여 텍스트에서 다음에 올 단어를 결정하는 것입니다.

- 아래와 같이 단계를 더 자세히 살펴보겠습니다 :

  1. **토큰화** :
     - LLM이 처리를 하기 전에 원시 입력 텍스트는 더 작은 단위(종종 하위 단어 또는 단어)로 토큰화 하여 모델이 인식할 수 있는 조각으로 입력을 나눕니다.
     - 모델에는 고정된 어휘목록(vocabulary)이 있습니다. 따라서, 토큰화 단계는 입력이 어휘목록과 일치하는 형식이 되도록 보장하기 때문에 매우 중요합니다.
     - GPT-3.5 및 GPT-4용 OpenAI 토크나이저는 [여기](https://platform.openai.com/tokenizer)에서 찾을 수 있습니다.
     - 자세한 내용은 [토큰화에 대한 입문서](https://aman.ai/primers/ai/tokenizer/)를 참조하세요.
       
  2. **임베딩** :
     - 각 토큰은 임베딩 매트릭스를 사용하여 고차원 벡터에 매핑됩니다. 이 벡터 표현은 토큰의 의미론적 의미를 포착하며 모델의 다음 레이어에 입력으로 사용됩니다.
     - 토큰의 순서에 대한 정보를 모델에 제공하기 위해 매핑된 임베딩에 위치 인코딩(positional encoding)이 추가됩니다. 이는 트랜스포머와 같은 모델이 고유한 순서 인식을 갖고 있지 않기 때문에 특히 중요합니다.

  3. **트랜스포머 구조** :
     - 대부분의 최신 LLM의 핵심은 트랜스포머 구조입니다.
     - 트랜스포머는 여러 레이어로 구성되어 있으며, 각 레이어에는 두 가지 주요 구성 요소가 있습니다 : multi-head self-attention 메커니즘과 position-wise feed-forward network 입니다.
     - 자기 어텐션 메커니즘은 각 토큰들이 자신과 관련해 중요성을 갖는 다른 토큰들에게 가중치를 부여할 수 있게 합니다. 이는 본질적으로 주어진 토큰과 관련있는 특정 부분에 대해 모델이 "주의를 기울일" 수 있도록 합니다.
       
     - After attention, the result is passed through a feed-forward neural network independently at each position.
     - Please refer to our primer on the Transformer architecture for more details.
       
  4. **Residual Connections** :
     - Each sub-layer (like self-attention or feed-forward neural network) in the model has a residual connection around it, followed by layer normalization. This helps in stabilizing the activations and speeds up training.

  5. **Output Layer** :
     - After passing through all the transformer layers, the final representation of each token is transformed into a vector of logits, where each logit corresponds to a word in the model’s vocabulary.
     - These logits describe the likelihood of each word being the next word in the sequence.

  6. **Probability Distribution** :
     - To convert the logits into probabilities, the Softmax function is applied. It normalizes the logits such that they all lie between 0 and 1 and sum up to 1.
     - The word with the highest probability can be chosen as the next word in the sequence.

  7. **Decoding** :
     - Depending on the application, different decoding strategies like greedy decoding, beam search, or top-k sampling might be employed to generate coherent and contextually relevant sequences.
     - Please refer to our primer on Token Sampling Methods for more details.

- Through this multi-step process, LLMs can generate human-like text, understand context, and provide relevant responses or completions to prompts.


### 2.1. LLM 학습 단계




### 2.2. Reasoning
