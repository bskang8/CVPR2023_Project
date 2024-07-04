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
     - 각 토큰은 임베딩 매트릭스를 사용하여 고차원 벡터에 매핑됩니다. 이 벡터 표현은 토큰의 맥락적 의미를 포착하며 모델의 다음 레이어에 입력으로 사용됩니다.
     - 토큰의 순서에 대한 정보를 모델에 제공하기 위해 매핑된 임베딩에 위치 인코딩(positional encoding)이 추가됩니다. 이는 트랜스포머와 같은 모델이 고유한 순서 인식을 갖고 있지 않기 때문에 특히 중요합니다.

  3. **트랜스포머 구조** :
     - 대부분의 최신 LLM의 핵심은 트랜스포머 구조입니다.
     - 트랜스포머는 여러 레이어로 구성되어 있으며, 각 레이어에는 두 가지 주요 구성 요소가 있습니다 : multi-head self-attention 메커니즘과 position-wise feed-forward network 입니다.
     - 자기 어텐션 메커니즘(self-attention mechanism)은 각 토큰들이 자신과 관련해 중요성을 갖는 다른 토큰들에게 가중치를 부여할 수 있게 합니다. 이는 본질적으로 주어진 토큰과 관련있는 특정 부분에 대해 모델이 "주의를 기울일" 수 있도록 합니다.
     - 어텐션 연산된 결과는, 각 위치에서 독립적으로 피드포워드 신경망으로 전달됩니다.
     - 자세한 내용은 [트랜스포머 아키텍처에 대한 입문서](https://aman.ai/primers/ai/transformers/)를 참조하세요.
       
  4. **잔차연결 (Residual Connection)** :
     - 모델의 각 하위 계층(예: 자기 어텐션 또는 피드포워드 신경망)은 주변에 잔여 연결이 적용된 후 계층 정규화가 수행됩니다. 이는 활성화를 안정화하고 훈련 속도를 높이는 데 도움이 됩니다.

  5. **출력 레이어** :
     - 모든 트랜스포머 레이어를 통과한 후, 각 토큰의 최종 표현은 모델의 어휘목록에 있는 각 단어에 대응하는 로짓 벡터로 변환됩니다. 
     - 이러한 로짓은 어휘 목록의 각 단어들이 시퀀스의 다음 단어가 될 가능성을 설명합니다.

  7. **확률분포** :
     - 로짓을 확률로 변환하기 위해 Softmax 함수가 적용됩니다. 이는 모두 0과 1 사이에 있고 합이 1이 되도록 로짓을 정규화합니다.
     - 어휘 목록의 단어들 중 확률이 가장 높은 단어가 시퀀스의 다음 단어로 선택될 수 있습니다.

  8. **디코딩 (Decoding)** :
     - 적용되는 상황에 따라 일관되고 문맥에 맞는 시퀀스를 생성하기 위하여, 그리디 디코딩(greedy decoding), 빔 검색(beam search), Top-K 샘플링(top-k sampling)과 같은 다양한 디코딩 전략이 사용됩니다.  
     - 자세한 내용은 [토큰 샘플링 방법](https://aman.ai/primers/ai/token-sampling/)에 대한 입문서를 참조하세요.

- 여러 단계의 프로세스를 통해, LLM은 인간과 유사한 텍스트를 생성하고, 맥락을 이해하고, 프롬프트에 대한 관련 응답이나 완성을 제공할 수 있습니다.

### 2.1. LLM 학습 단계

- 상위 수준에서, LLMs의 훈련에 포함되는 단계는 다음과 같습니다:
    1. **문서(코퍼스, corpus) 준비** : 뉴스 기사, 소셜 미디어 게시물, 웹 문서 등 대규모 텍스트 데이터 모음을 수집합니다.
    2. **토큰화** : 텍스트를 토큰이라고 하는 개별 단어 또는 하위 단어로 분할합니다.
    3. **임베딩 생성** : 일반적으로 훈련을 처음 시작할 때 PyTorch의 nn.Embedding 클래스를 통해 랜덤하게 초기화된 임베딩 테이블을 사용합니다. 또한, Word2Vec, GloVe, FastText 등과 같은 사전 훈련된 임베딩도 사용할 수 있습니다. 이러한 임베딩은 입력 토큰의 맥락화되지 않은 벡터 형식을 나타냅니다.
    4. **신경망 훈련** : 입력 토큰에 대한 신경망 모델을 훈련합니다.
         - BERT 및 그 변형과 같은 인코더 모델의 경우 모델은 마스킹된 특정 단어의 전후 맥락(주변 단어)을 예측하는 방법을 학습합니다.
         - BERT는 특히 마스킹된 단어를 예측하는 마스크드 언어 모델링 작업(Masked Language Modeling task 또는 Cloze task)과 다음 문장 예측 작업으로 훈련되었습니다; [BERT 입문서](https://aman.ai/primers/ai/bert/)에 설명되어 있습니다.
         - GPT-N, LLaMA 등과 같은 디코더 모델의 경우 주어진 이전 토큰들의 맥락을 고려하여 시퀀스의 다음 토큰을 예측하는 방법을 학습합니다.

### 2.2. Reasoning

- Let’s delve into how reasoning works in LLMs; we will define reasoning as the “ability to make inferences using evidence and logic.” (source)
- There are a multitude of varieties of reasoning, such as commonsense reasoning or mathematical reasoning.
- Similarly, there are a variety of methods to elicit reasoning from the model, one of them being chain-of-thought prompting which can be found here.
- It’s important to note that the extent of how much reasoning an LLM uses in order to give its final prediction is still unknown, since teasing apart the contribution of reasoning and factual information to derive the final output is not a straightforward task.
