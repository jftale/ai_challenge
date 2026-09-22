# 베이스라인 v3 → 개선판, 단계별 수정 가이드

원본 베이스라인 코드를 **직접 고쳐가며** 점수를 올리는 방법입니다.
코딩을 잘 몰라도 따라올 수 있도록, **어느 셀의 어느 줄을 무엇으로 바꾸는지** 하나씩 적었습니다.

> **더 빠른 길:** 그냥 `vqa_qwen25vl_solution.ipynb`를 쓰시면 아래가 전부 적용되어 있습니다.
> 하지만 **직접 고쳐봐야 이해가 됩니다.** 그리고 대회 발표 때 "무엇을 왜 바꿨는지"를
> 설명할 수 있어야 하니, 한 번은 손으로 해보시길 권합니다.

---

## 전체 그림

**총 10단계**입니다. 위에서부터 순서대로 하면 되고, **아무 데서나 멈춰도 됩니다.**
각 단계는 앞 단계에 의존하지만, 뒤 단계를 안 해도 코드는 정상 동작합니다.

### 1부 — 학습은 손대지 않고, 추론만 고치기 (1~6단계)

| 단계 | 무엇을 | 효과 | 난이도 | 작업시간 |
|---|---|---|---|---|
| 1 | 해상도 올리기 | ⭐⭐⭐⭐⭐ | 아주 쉬움 | 1분 |
| 2 | 사진 회전 보정 | ⭐⭐ | 아주 쉬움 | 2분 |
| 3 | 프롬프트 다듬기 | ⭐ | 아주 쉬움 | 2분 |
| 4 | 생성 → 확률 비교로 교체 | ⭐⭐⭐⭐ | 보통 | 15분 |
| 5 | 보기 순서 TTA | ⭐⭐⭐ | 보통 | 15분 |
| 6 | 모델 교체 | ⭐⭐⭐⭐ | 쉬움 | 5분 |

> **1부만 해도 베이스라인보다 크게 오릅니다.** 파인튜닝은 건드리지 않았으므로
> 학습 시간(30분+)도 필요 없습니다. 시간이 없다면 여기서 멈추세요.

### 2부 — 학습까지 고치기 (7~10단계)

| 단계 | 무엇을 | 효과 | 난이도 | 작업시간 |
|---|---|---|---|---|
| 7 | 라벨 마스킹 (버그 수정) | ⭐⭐⭐ | 보통 | 15분 |
| 8 | 전체 데이터 + 보기 셔플 증강 | ⭐⭐ | 쉬움 | 10분 |
| 9 | 검증 기준을 loss → 정확도로 | ⭐⭐ | 보통 | 20분 |
| 10 | 제로샷과 비교해서 더 나은 쪽 채택 | ⭐⭐ (과적합 방어) | 쉬움 | 10분 |

---
---

# 1부 — 추론만 고치기

## 1단계. 해상도 올리기 ⭐⭐⭐⭐⭐

**가장 중요합니다. 이거 하나만 해도 체감됩니다.**

### 왜?

이 대회는 사실상 **이미지 속 글자를 읽는 문제**입니다.
그런데 베이스라인은 이미지를 **448×448 픽셀**로 줄여버립니다.
간판 글씨는 이 시점에 이미 뭉개져서 사라집니다.

**읽을 수 없는 정보는 아무리 학습해도 맞힐 수 없습니다.**

VLM은 이미지를 `28×28` 픽셀 조각 = **비전 토큰 1개**로 잘라서 봅니다.
`max_pixels = 256 * 28 * 28`은 "토큰을 최대 256개까지만 쓰겠다"는 뜻이고,
256토큰은 약 448×448 픽셀입니다.

### 어디를?

**"모델, Processor" 셀**의 `processor = AutoProcessor.from_pretrained(...)` 부분

### 바꾸기 전

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=128 * 28 * 28,
    max_pixels=256 * 28 * 28,     # ← 약 448×448. 너무 작다
    trust_remote_code=True,
)
```

### 바꾼 후

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=64 * 28 * 28,
    max_pixels=1024 * 28 * 28,    # ← 약 896×896. 글씨가 살아난다
    trust_remote_code=True,
)
```

### 주의

- 학습 중 **`CUDA out of memory`** 가 뜨면 `1024`를 `512`나 `640`으로 낮추세요.
- **추론 때만 더 올릴 수도 있습니다.** 추론 셀 맨 앞에 이 한 줄을 넣으면 됩니다:

```python
processor.image_processor.max_pixels = 1280 * 28 * 28   # 추론은 학습보다 메모리 여유가 있음
```

---

## 2단계. 사진 회전 보정 ⭐⭐

### 왜?

휴대폰 사진은 **실제 픽셀은 가로인데, "세로로 돌려서 보여줘"라는 메모(EXIF)만 붙어 있는** 경우가 많습니다.
`Image.open()`만 하면 그 메모를 무시해서 **글자가 90도 누운 채로** 모델에 들어갑니다.
누운 글자는 모델이 훨씬 못 읽습니다.

### 어디를?

`Image.open(...)` 이 나오는 **모든 곳** (보통 2군데: Dataset 안, 추론 루프 안)

### 바꾸기 전

```python
from PIL import Image

img = Image.open(row["path"]).convert("RGB")
```

### 바꾼 후

```python
from PIL import Image, ImageOps          # ← ImageOps 추가

img = Image.open(row["path"])
img = ImageOps.exif_transpose(img)       # ← 이 한 줄이 회전을 바로잡음
img = img.convert("RGB")
```

매번 3줄 쓰기 번거로우니, **함수로 한 번만 만들어두고** 재사용하는 걸 권합니다:

```python
from PIL import Image, ImageOps
Image.MAX_IMAGE_PIXELS = None

def load_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")
```

그리고 기존 `Image.open(row["path"]).convert("RGB")` 를 전부 `load_image(row["path"])` 로 바꾸세요.

---

## 3단계. 프롬프트 다듬기 ⭐

### 왜?

효과는 작지만 **2분이면 끝나고 손해가 없습니다.**
데이터가 한국어이므로 지시문도 한국어로 주고, "글자를 꼼꼼히 보라"고 명시해줍니다.

### 어디를?

**"프롬프트 템플릿" 셀** 전체

### 바꾸기 전

```python
SYSTEM_INSTRUCT = (
    "You are a helpful visual question answering assistant. "
    "Answer using exactly one letter among a, b, c, or d. No explanation."
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n"
        f"(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n\n"
        "정답을 반드시 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )
```

### 바꾼 후

```python
SYSTEM_INSTRUCT = (
    "당신은 이미지 속의 글자와 장면을 아주 꼼꼼하게 읽어내는 한국어 시각 질의응답(VQA) 전문가입니다. "
    "간판, 메뉴판, 표지판, 안내문에 적힌 작은 글씨까지 정확히 확인한 뒤 답합니다. "
    "반드시 a, b, c, d 중 소문자 한 글자만 출력합니다."
)

def build_mc_prompt(question, a, b, c, d):
    return (
        f"질문: {question}\n"
        f"\n보기:\n(a) {a}\n(b) {b}\n(c) {c}\n(d) {d}\n"
        f"\n이미지에 실제로 적혀 있는 글자와 세부 정보를 근거로, 위 보기 중 정답 하나를 고르세요.\n"
        f"출력은 오직 a, b, c, d 중 소문자 한 글자입니다. 설명은 쓰지 마세요.\n"
        f"답:"
    )
```

---

## 4단계. 생성 → 확률 비교로 교체 ⭐⭐⭐⭐

**두 번째로 중요한 단계입니다.**

### 왜?

베이스라인은 모델이 글자를 **생성**하게 한 다음, 그 문자열에서 `a~d`를 찾습니다.

```python
def extract_choice(text):
    ...
    return "a"        # ← 못 찾으면 무조건 "a"
```

모델이 `"정답은"`, 공백, 한글 같은 걸 뱉는 순간 **찍기가 됩니다.**

대신 **`a`, `b`, `c`, `d` 네 글자의 확률을 직접 꺼내서 비교**하면:

- **파싱 실패가 원천적으로 불가능** (항상 4개 중에서만 고름)
- 토큰 1개만 계산하므로 **더 빠름**
- 확률이 남으므로 5단계 TTA에서 그대로 쓸 수 있음

### 어디를?

**"inference" 셀** 전체를 아래로 갈아끼웁니다.
`extract_choice` 함수는 **더 이상 필요 없습니다. 지워도 됩니다.**

### 바꾼 후 (셀 전체 교체)

```python
import numpy as np
import torch
from tqdm.auto import tqdm

LETTERS = ["a", "b", "c", "d"]
tok = processor.tokenizer

# a/b/c/d 를 뜻하는 토큰 번호를 미리 찾아둔다.
# 모델이 대문자 "A"를 뱉을 수도 있으므로 대소문자를 함께 묶어 확률을 합친다.
CHOICE_IDS = [
    sorted({tok.encode(v, add_special_tokens=False)[0] for v in (L, L.upper())})
    for L in LETTERS
]
print("선지 토큰:", dict(zip(LETTERS, CHOICE_IDS)))


def score_one(img, question, options):
    """이미지 1장 + 보기 4개 -> [4] 확률 배열을 돌려준다"""
    user_text = build_mc_prompt(question, *options)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1,             # 딱 1토큰만
            do_sample=False,
            repetition_penalty=1.0,       # 기본 설정이 확률을 왜곡하지 않도록
            output_logits=True,           # ★ 가공 안 된 원본 확률을 받는다
            return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id,
        )

    logits = out.logits[0][0].float()                     # [단어사전 크기]
    logprobs = torch.log_softmax(logits, dim=-1)
    # a/b/c/d 네 개만 뽑아서 다시 정규화 -> 합이 1인 4지선다 확률
    per_letter = torch.stack([torch.logsumexp(logprobs[ids], dim=-1) for ids in CHOICE_IDS])
    return torch.softmax(per_letter, dim=-1).cpu().numpy()


# ── 추론 ──────────────────────────────────────────────
model.eval()
model.config.use_cache = True
try:
    model.gradient_checkpointing_disable()   # 학습용 설정을 꺼야 추론이 빠름
except Exception:
    pass

preds = []
for i in tqdm(range(len(test_df)), desc="Inference", unit="sample"):
    row = test_df.iloc[i]
    img = load_image(row["path"])                    # 2단계에서 만든 함수
    options = [row["a"], row["b"], row["c"], row["d"]]
    p = score_one(img, row["question"], options)
    preds.append(LETTERS[p.argmax()])

submission = pd.DataFrame({"id": test_df["id"], "answer": preds})
submission.to_csv("/content/submission.csv", index=False)
print("Saved /content/submission.csv")
print(submission["answer"].value_counts())
```

### 확인 방법

`print(submission["answer"].value_counts())` 결과가 **a에 심하게 쏠려 있지 않으면** 잘 된 겁니다.
베이스라인은 파싱 실패 때문에 `a`가 비정상적으로 많이 나오는 경우가 흔합니다.

---

## 5단계. 보기 순서 TTA ⭐⭐⭐

### 왜?

언어모델에는 **"잘 모르겠으면 (a)를 고른다"는 버릇(위치 편향)** 이 있습니다.

같은 문제를 **보기 순서만 바꿔서 4번** 물어보고, **내용 기준으로** 확률을 평균내면 이 버릇이 사라집니다.

```
1회차:  (a)=1번보기  (b)=2번  (c)=3번  (d)=4번
2회차:  (a)=2번보기  (b)=3번  (c)=4번  (d)=1번
3회차:  (a)=3번보기  (b)=4번  (c)=1번  (d)=2번
4회차:  (a)=4번보기  (b)=1번  (c)=2번  (d)=3번
```

각 보기가 a/b/c/d 자리에 **정확히 한 번씩** 오므로 위치 효과가 완전히 상쇄됩니다.
**시간은 4배**가 되지만, 공짜로 얻는 정확도 상승폭이 큽니다.

### 어디를?

4단계에서 만든 추론 루프의 `for i in tqdm(...)` 안쪽

### 바꾸기 전

```python
for i in tqdm(range(len(test_df)), desc="Inference", unit="sample"):
    row = test_df.iloc[i]
    img = load_image(row["path"])
    options = [row["a"], row["b"], row["c"], row["d"]]
    p = score_one(img, row["question"], options)
    preds.append(LETTERS[p.argmax()])
```

### 바꾼 후

```python
N_PERM = 4        # 1로 두면 TTA를 끄는 것. 시간이 없으면 2도 괜찮습니다.

for i in tqdm(range(len(test_df)), desc="Inference", unit="sample"):
    row = test_df.iloc[i]
    img = load_image(row["path"])
    options = [row["a"], row["b"], row["c"], row["d"]]

    probs = np.zeros(4)
    for s in range(N_PERM):
        # perm[k] = "화면의 k번째 자리에 놓을, 원래 보기의 번호"
        perm = [(k + s) % 4 for k in range(4)]
        shown = [options[perm[k]] for k in range(4)]

        p = score_one(img, row["question"], shown)

        # 화면 자리 기준 확률을 -> 원래 보기 기준으로 되돌려서 누적
        for k in range(4):
            probs[perm[k]] += p[k]

    preds.append(LETTERS[probs.argmax()])
```

### 헷갈리기 쉬운 부분

`probs[perm[k]] += p[k]` 이 한 줄이 핵심입니다.

모델은 **"화면에 보이는 (a)(b)(c)(d)"** 기준으로 확률을 줍니다.
그런데 우리가 알고 싶은 건 **"원래 CSV의 a/b/c/d"** 입니다.
`perm[k]`가 그 둘을 이어주는 번역표입니다.

예를 들어 `s=1`일 때 `perm = [1,2,3,0]`이므로,
화면 `(a)`자리(k=0)에는 원래 보기 1번(=`b`)이 놓여 있습니다.
그러니 화면 `(a)`의 확률 `p[0]`은 **원래 `b`의 점수**로 더해야 맞습니다.

---

## 6단계. 모델 교체 ⭐⭐⭐⭐

### 왜?

Qwen3-VL은 Qwen2.5-VL 대비 이런 점들이 좋아졌습니다:

- OCR 지원 언어 **10개 → 39개**
- **어두운 사진, 흔들린 사진, 기울어진 사진에 강함**
- 이미지 정보를 LLM의 **여러 층에 나눠 주입**(DeepStack)해서 작은 디테일을 더 잘 봄

휴대폰으로 찍은 간판 사진이 정확히 이런 조건입니다. **크기보다 이 내용이 중요합니다.**

### 어디를? (3군데)

**(1) 설치 셀** — Qwen3-VL은 transformers 4.57 이상이 필요합니다

```python
# 바꾸기 전
!pip -q install git+https://github.com/huggingface/transformers accelerate

# 바꾼 후 (GitHub 최신 소스는 날짜에 따라 깨지므로 릴리스 버전을 씁니다)
!pip -q install -U "transformers>=4.57.0" "accelerate>=0.34.0" "peft>=0.13.2" "bitsandbytes>=0.43.0"
```

설치 후 **런타임 → 세션 다시 시작**을 꼭 해주세요.

**(2) "라이브러리, 데이터, 설정" 셀 — 묶음 import 안에서 클래스 이름만 교체**

> ⚠️ **묶음 전체를 지우면 안 됩니다.** 나머지 3개도 뒤에서 쓰이므로,
> 지우면 `NameError: name 'AutoProcessor' is not defined` 같은 에러가 납니다.
> **첫 줄 한 개만** 바꾸세요.

```python
# 바꾸기 전
from transformers import (
    Qwen2_5_VLForConditionalGeneration,     # ← 이 줄만 교체
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
```

```python
# 바꾼 후
from transformers import (
    AutoModelForImageTextToText,            # ← 교체됨. 나머지 3개는 그대로!
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
```

**(3) 같은 셀 — 모델 이름과 픽셀 단위**

```python
# 바꾸기 전
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
```

```python
# 바꾼 후
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
PX = 32 if "Qwen3-VL" in MODEL_ID else 28   # 토큰 1개가 담는 픽셀 (모델마다 다름)

# 버전이 낮으면 여기서 바로 잡아줍니다 (런타임 재시작을 깜빡하는 실수 방지)
import transformers
print("transformers:", transformers.__version__)
assert tuple(int(x) for x in transformers.__version__.split(".")[:2]) >= (4, 57), (
    "Qwen3-VL은 transformers 4.57 이상이 필요합니다. "
    "설치 후 [런타임 → 세션 다시 시작]을 했는지 확인하세요."
)
```

**(4) "모델, Processor" 셀 — 클래스 이름 교체**

`from_pretrained`의 **인자는 하나도 바꿀 필요가 없습니다.** 클래스 이름만 바뀝니다.

```python
# 바꾸기 전
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
```

```python
# 바꾼 후
base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
```

> 💡 `AutoModelForImageTextToText`는 **Qwen2.5-VL도 그대로 열립니다.**
> 그래서 모델을 아직 안 바꿀 생각이어도 (2)(4)의 클래스 교체는 **지금 해둬도 안전**합니다.
> 나중에 `MODEL_ID` 한 줄만 바꾸면 모델이 교체되는 구조가 됩니다.

### ⚠️ 여기서 가장 많이 실수합니다

**Qwen3-VL은 토큰 1개가 `32×32` 픽셀입니다. Qwen2.5-VL은 `28×28`이었습니다.**

`28`을 그대로 두고 모델만 바꾸면 **해상도가 23% 손해인 채로 조용히 돌아갑니다.**
에러도 안 나기 때문에 알아채기 어렵습니다.

```python
# Qwen3-VL을 쓴다면 28 → 32 로 전부 바꾸세요
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=64 * 32 * 32,        # ← 28이 아니라 32
    max_pixels=1024 * 32 * 32,      # ← 28이 아니라 32
    trust_remote_code=True,
)
```

(3)에서 `PX`를 이미 만들어뒀으므로, processor는 이렇게 쓰면 모델에 따라 자동으로 맞춰집니다:

```python
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=64 * PX * PX,        # 숫자 28/32를 직접 쓰지 않는 게 핵심
    max_pixels=1024 * PX * PX,
    trust_remote_code=True,
)
```

### 메모리가 부족하면

| GPU | 추천 |
|---|---|
| A100 40GB+ | `Qwen/Qwen3-VL-8B-Instruct`, 양자화 없이 |
| L4 / 3090 / 4090 (24GB) | `Qwen/Qwen3-VL-8B-Instruct`, 4bit |
| T4 / 5060Ti (16GB) | `Qwen/Qwen3-VL-8B-Instruct`, 4bit, `max_pixels`를 `640`으로 |
| 12GB 이하 | `Qwen/Qwen3-VL-4B-Instruct`, 4bit |

---
---

# 2부 — 학습까지 고치기

> 1부까지만 해도 충분히 좋습니다. **여기부터는 시간이 있을 때** 하세요.

## 7단계. 라벨 마스킹 ⭐⭐⭐ (버그 수정)

### 왜?

베이스라인의 이 한 줄이 문제입니다:

```python
enc["labels"] = enc["input_ids"].clone()
```

`labels`는 **"모델이 맞춰야 할 정답"** 입니다.
그런데 `input_ids` 전체를 넣으면, 모델은 이런 것까지 전부 "따라 쓰도록" 학습합니다:

- 시스템 프롬프트
- 질문 문장
- 보기 4개
- 이미지 토큰
- **패딩(빈칸 채우기용 쓰레기 토큰)**

결과적으로 **정답 고르는 능력은 거의 안 늘고, 질문 문장만 외워서 과적합**됩니다.

우리가 학습시키고 싶은 건 오직 **정답 글자 하나**입니다.

### 어디를?

**"Custom Dataset, Collator" 셀**의 `DataCollator` 클래스

### 바꾸기 전

```python
        if self.train:
            enc["labels"] = enc["input_ids"].clone()

        return enc
```

### 바꾼 후

셀 맨 위에 이 함수를 추가하고:

```python
# "<|im_start|>assistant\n" = "이제부터 모델이 답할 차례" 라는 표식
HDR = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
print("assistant 표식 토큰:", HDR)


def make_labels(input_ids, attention_mask):
    """assistant 표식 뒤(= 정답 글자)만 학습 대상으로 남기고 나머지는 -100"""
    labels = torch.full_like(input_ids, -100)      # -100 = "이 자리는 학습하지 마라"
    H = len(HDR)
    hdr = torch.tensor(HDR, dtype=input_ids.dtype)

    for i in range(input_ids.size(0)):
        ids = input_ids[i]
        # 뒤에서부터 표식을 찾는다 (질문 안에 비슷한 게 있어도 마지막 것이 진짜)
        for j in range(ids.size(0) - H, -1, -1):
            if torch.equal(ids[j:j + H], hdr):
                labels[i, j + H:] = ids[j + H:]    # 표식 뒤부터만 학습
                break

    labels[attention_mask == 0] = -100             # 패딩 제외
    return labels
```

`DataCollator` 안을 이렇게 바꿉니다:

```python
        if self.train:
            enc["labels"] = make_labels(enc["input_ids"], enc["attention_mask"])

        return enc
```

### 반드시 확인하세요

바꾼 뒤 아래를 실행해서 **정답 글자만 학습되는지** 눈으로 확인하세요.
표식을 못 찾으면 **전부 -100이 되어 학습이 조용히 무의미해집니다.**

```python
_ds = VQAMCDataset(train_df.head(2), processor, train=True)
_b = DataCollator(processor, True)([_ds[0], _ds[1]])

_n = int((_b["labels"][0] != -100).sum())
_sup = _b["labels"][0][_b["labels"][0] != -100]

print("전체 토큰:", _b["input_ids"].shape[1], "| 학습 대상 토큰:", _n)
print("학습 대상 내용:", repr(processor.tokenizer.decode(_sup)))

assert _n > 0,  "표식을 못 찾았습니다! 학습이 무의미해집니다."
assert _n <= 8, "학습 대상이 너무 많습니다. 프롬프트까지 학습될 위험이 있습니다."
print("✅ 정상")
```

`학습 대상 내용: 'a<|im_end|>\n'` 처럼 **정답 글자로 시작하면 성공**입니다.

---

## 8단계. 전체 데이터 + 보기 셔플 증강 ⭐⭐

### (1) 1000개 제한 풀기

**어디를?** "라이브러리, 데이터, 설정" 셀

```python
# 바꾸기 전 — 데이터를 버릴 이유가 없습니다
train_df = train_df.sample(n=min(1000, len(train_df)), random_state=SEED).reset_index(drop=True)

# 바꾼 후
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)   # 섞기만 하고 전부 사용
```

> **주의:** T4에서 전체(수천 건)를 돌리면 몇 시간이 걸립니다.
> 시간이 부족하면 `n=1000` 대신 `n=4000` 정도로 **늘리는** 선에서 타협하세요.

### (2) 보기 순서 셔플 증강

**왜?** 매 학습마다 보기 순서를 섞으면 **데이터가 실질적으로 4배**가 되고,
"정답은 주로 a"같은 편향을 외우는 걸 막습니다. 5단계 TTA와 짝을 이룹니다.

**어디를?** `VQAMCDataset`의 `__getitem__`

```python
# 바꾸기 전
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")

        q = str(row["question"])
        a, b, c, d = str(row["a"]), str(row["b"]), str(row["c"]), str(row["d"])
        user_text = build_mc_prompt(q, a, b, c, d)

        # (messages 만드는 부분은 그대로 — 생략)

        if self.train:
            gold = str(row["answer"]).strip().lower()
```

```python
# 바꾼 후
    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_image(row["path"])              # 2단계 함수

        q = str(row["question"])
        options = [str(row["a"]), str(row["b"]), str(row["c"]), str(row["d"])]
        gold_idx = ["a", "b", "c", "d"].index(str(row["answer"]).strip().lower())

        if self.train:
            # 보기 순서를 섞고, 정답 글자도 따라서 바꾼다
            # (random은 베이스라인 첫 셀에서 이미 import 되어 있습니다)
            rng = random.Random(SEED * 1000003 + i)
            perm = list(range(4))
            rng.shuffle(perm)
            options = [options[perm[k]] for k in range(4)]
            gold_idx = perm.index(gold_idx)        # ★ 정답 위치도 함께 이동

        user_text = build_mc_prompt(q, *options)

        # (messages 만드는 부분은 그대로 — 생략)

        if self.train:
            gold = ["a", "b", "c", "d"][gold_idx]
```

### ⚠️ 가장 흔한 실수

**보기만 섞고 정답 글자를 안 바꾸면, 전부 틀린 라벨로 학습합니다.**
`gold_idx = perm.index(gold_idx)` 이 줄을 절대 빠뜨리지 마세요.

확인하는 법:

```python
_ds = VQAMCDataset(train_df.head(1), processor, train=True)
s = _ds[0]
print("정답으로 학습되는 글자:", s["messages"][-1]["content"][0]["text"])
print("그 글자가 가리키는 보기:", s["messages"][1]["content"][1]["text"])
# 원본 정답 내용과 같은지 눈으로 대조
print("원본 정답 내용:", train_df.iloc[0][train_df.iloc[0]["answer"]])
```

---

## 9단계. 검증 기준을 loss → 정확도로 ⭐⭐

### 왜?

베이스라인은 `valid loss`만 출력합니다. 그런데 **우리가 올리려는 건 정확도**입니다.
**loss는 내려가는데 정확도는 떨어지는 구간이 실제로 존재합니다.**

또 지금 구조로는 "학습이 끝난 마지막 상태"를 그냥 쓰는데,
중간에 더 좋았던 시점이 있었다면 그걸 놓칩니다.

### 어디를?

학습 루프의 검증 부분

### 바꾸기 전

```python
    # 검증
    model.eval()
    val_loss_sum = 0.0
    with torch.no_grad():
        for vb in tqdm(valid_loader, ...):
            ...
            val_loss_sum += val_outputs.loss.item()
    avg_val_loss = val_loss_sum / len(valid_loader)
    print(f"[Epoch {epoch + 1}] valid loss: {avg_val_loss:.4f}")
```

### 바꾼 후

**(A) 학습 루프 셀 맨 위**에 함수와 변수를 추가합니다. 들여쓰기 없이 맨 왼쪽부터 씁니다.

```python
from peft import get_peft_model_state_dict, set_peft_model_state_dict

best_acc, best_state = -1.0, None      # 지금까지 가장 좋았던 기록


def eval_accuracy(df):
    """검증 세트 정확도를 잰다 (4단계에서 만든 score_one 재사용)"""
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = True

    correct = 0
    for i in range(len(df)):
        row = df.iloc[i]
        img = load_image(row["path"])
        options = [row["a"], row["b"], row["c"], row["d"]]
        p = score_one(img, row["question"], options)
        if LETTERS[p.argmax()] == str(row["answer"]).strip().lower():
            correct += 1

    # 학습 설정으로 되돌리기 (★ 안 하면 다음 스텝에서 메모리 폭증)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    return correct / len(df)
```

**(B) 기존 검증 부분을 삭제**하고 그 자리에 아래를 넣습니다.
`for epoch in ...` 루프 **안**이므로, 원래 검증 코드와 **같은 들여쓰기**를 맞춰야 합니다
(보통 공백 4칸).

```python
    # ↓ 이 블록 전체가 for epoch 루프 안에 들어갑니다 (공백 4칸 들여쓰기)
    acc = eval_accuracy(valid_subset)
    print(f"[Epoch {epoch + 1}] 검증 정확도: {acc*100:.2f}%")

    if acc > best_acc:
        best_acc = acc
        best_state = {k: v.detach().cpu().clone()
                      for k, v in get_peft_model_state_dict(model).items()}
        print("  ⭐ 최고 기록 갱신 — 이 시점을 저장")
```

**(C) 학습 루프가 전부 끝난 뒤** (들여쓰기 없이 맨 왼쪽부터):

```python
if best_state is not None:
    set_peft_model_state_dict(model, {k: v.to(model.device) for k, v in best_state.items()})
    print(f"✅ 가장 좋았던 시점으로 되돌렸습니다 (정확도 {best_acc*100:.2f}%)")
```

> **들여쓰기 주의:** 파이썬은 들여쓰기로 "어디에 속하는 코드인지"를 판단합니다.
> (B)는 반드시 `for epoch` 루프 안쪽, (A)와 (C)는 맨 왼쪽이어야 합니다.
> 잘못하면 `IndentationError` 가 납니다.

### ⚠️ 놓치기 쉬운 함정

`eval_accuracy` 안에서 `gradient_checkpointing_disable()`을 했으면
**반드시 끝에서 다시 켜야 합니다.** 안 켜면 다음 학습 스텝부터 메모리가 급증해
**T4에서 OOM이 납니다.** 위 코드에 이미 포함해뒀습니다.

### 속도 주의

검증 1회가 수 분 걸릴 수 있습니다. 검증 세트가 크면 앞부분만 쓰세요:

```python
valid_small = valid_subset.head(200).reset_index(drop=True)
acc = eval_accuracy(valid_small)
```

---

## 10단계. 제로샷과 비교해서 더 나은 쪽 채택 ⭐⭐

**과적합을 막는 마지막 안전장치입니다.**

### 왜?

**파인튜닝이 항상 도움이 되는 건 아닙니다.** 데이터가 적거나 에폭이 많으면 오히려 나빠집니다.

그런데 보통은 그걸 **모르는 채로 제출**합니다.
그러지 말고, **검증 세트로 직접 비교해서 이긴 쪽을 쓰면 됩니다.**

LoRA는 껐다 켤 수 있으므로 (`model.disable_adapter()`), 모델을 두 번 로드할 필요도 없습니다.

### 어디를?

학습이 끝난 뒤, 추론 셀 **직전**에 추가

```python
# 1) 파인튜닝 끈 상태(= 원래 사전학습 모델)의 성능
with model.disable_adapter():
    acc_zeroshot = eval_accuracy(valid_small)

# 2) 파인튜닝 켠 상태의 성능
acc_finetuned = eval_accuracy(valid_small)

print(f"제로샷   : {acc_zeroshot*100:.2f}%")
print(f"파인튜닝 : {acc_finetuned*100:.2f}%")

# 3) 0.5%p 이상 확실히 이길 때만 파인튜닝을 쓴다
#    (검증 200문항이면 ±3%p 정도는 그냥 운입니다. 박빙이면 단순한 쪽이 안전합니다.)
USE_FINETUNED = acc_finetuned > acc_zeroshot + 0.005

if USE_FINETUNED:
    print("→ 파인튜닝 모델로 제출합니다")
else:
    print("→ 파인튜닝이 도움이 안 됐습니다. 제로샷으로 제출합니다")
```

그리고 추론 루프를 이렇게 감쌉니다:

```python
import contextlib

# USE_FINETUNED가 False면 LoRA를 끈 채로 추론
ctx = contextlib.nullcontext() if USE_FINETUNED else model.disable_adapter()

with ctx:
    preds = []
    for i in tqdm(range(len(test_df)), desc="Inference", unit="sample"):
        ...   # 5단계의 추론 루프를 그대로
```

---
---

# 문제가 생겼을 때

### `CUDA out of memory`

순서대로 시도하세요:

1. `max_pixels`를 낮춘다 (`1024` → `640` → `512`)
2. 모델을 작게 (`Qwen3-VL-8B` → `Qwen3-VL-4B`)
3. **런타임 → 세션 다시 시작** 후 처음부터 (메모리가 조각나 있을 수 있음)
4. 학습 중이라면 `GRAD_ACCUM`을 늘리고 배치는 1 유지

### 추론이 너무 오래 걸린다

- `N_PERM = 4` → `2` (시간 절반, 정확도 1~3%p 손해)
- 파인튜닝을 아예 건너뛰기 (1부만 적용) — **이게 가장 효율적입니다**

### `Qwen3VLForConditionalGeneration is not supported` 같은 에러

`transformers` 버전이 낮습니다.

```python
!pip -q install -U "transformers>=4.57.0"
```

설치 후 **런타임 → 세션 다시 시작**을 꼭 하세요. 안 하면 옛 버전이 그대로 메모리에 남아 있습니다.

### 점수가 오히려 떨어졌다

한 단계씩 되돌리면서 범인을 찾으세요. 경험상 흔한 원인:

- **8단계에서 보기는 섞었는데 정답 글자를 안 바꿈** (가장 흔함 — 8단계 확인 코드 실행)
- **6단계에서 모델은 바꿨는데 `28`을 `32`로 안 바꿈** (해상도 손실)
- **7단계 검사를 건너뛰어서 라벨이 전부 -100** (학습이 무의미)

---

# 마지막으로

### 우선순위 요약

시간이 없다면 **1, 4, 5, 6번만** 하세요. 이 네 개가 효과의 대부분입니다.
전부 추론 쪽이라 **학습을 한 번도 안 돌려도 됩니다.**

### 발표 준비용

각 단계마다 **바꾸기 전/후 점수를 리더보드에 찍어서 기록해두세요.**
"무엇을 왜 바꿨고 몇 점이 올랐는지"가 그대로 발표 자료가 됩니다.
Discussion 점수도 여기서 나옵니다.

### 이 가이드의 한계

여기 적힌 **효과(⭐ 개수)는 설계 근거이지 실측값이 아닙니다.**
실제 상승폭은 본인 데이터와 GPU에서 직접 확인해야 합니다.
9~10단계의 검증 장치가 정확히 그 용도입니다.
