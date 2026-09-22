# -*- coding: utf-8 -*-
"""Extract and unit-test the pure-logic pieces of the notebook (no GPU/torch needed)."""
import json, re, difflib, random
import numpy as np
from collections import Counter

LETTERS = ["a", "b", "c", "d"]
ok = lambda m: print("  PASS", m)

# ───────────────────────────────────────────────────────────
# 1. Permutation TTA round-trip
# ───────────────────────────────────────────────────────────
print("[1] 보기 순환 치환 TTA 매핑")
n_perm = 4
perms = [[(k + s) % 4 for k in range(4)] for s in range(n_perm)]

# each original option must appear exactly once in every display slot
slot_counts = np.zeros((4, 4), int)   # [display_slot][original_idx]
for perm in perms:
    for k in range(4):
        slot_counts[k][perm[k]] += 1
assert (slot_counts == 1).all(), slot_counts
ok("각 보기가 a/b/c/d 자리에 정확히 1번씩 등장 (위치 편향 완전 상쇄)")

# simulate: model is perfect on content #2, plus a strong "prefers slot a" bias
options = ["역삼", "선릉", "사상", "강남"]
TRUE = 2
probs = np.zeros(4)
for perm in perms:
    shown = [options[perm[k]] for k in range(4)]
    p = np.array([0.25] * 4)
    p[shown.index(options[TRUE])] += 0.30      # real signal
    p[0] += 0.45                               # positional bias toward slot (a) — 신호보다 강하게
    p = p / p.sum()
    for k in range(4):
        probs[perm[k]] += p[k]                 # map display slot -> original option
probs /= probs.sum()
assert probs.argmax() == TRUE, probs
ok(f"위치 편향(+0.45)이 신호(+0.30)보다 강해도 정답 복원: {np.round(probs,3)} -> {LETTERS[probs.argmax()]}")

# without TTA the same bias flips the answer to 'a' -> proves TTA is doing work
p1 = np.array([0.25] * 4); p1[TRUE] += 0.30; p1[0] += 0.45; p1 /= p1.sum()
assert p1.argmax() == 0
ok(f"TTA 1회만 하면 편향에 져서 오답 'a': {np.round(p1,3)}")

# ───────────────────────────────────────────────────────────
# 2. build_labels (numpy re-implementation, identical algorithm)
# ───────────────────────────────────────────────────────────
print("\n[2] 라벨 마스킹 (assistant 응답만 학습)")
HDR = [1001, 1002, 1003]        # <|im_start|> assistant \n
def build_labels_np(input_ids, attention_mask, hdr=HDR):
    labels = np.full_like(input_ids, -100)
    H = len(hdr)
    for i in range(input_ids.shape[0]):
        ids = input_ids[i]; start = -1
        for j in range(ids.shape[0] - H, -1, -1):
            if list(ids[j:j+H]) == list(hdr):
                start = j + H; break
        if start >= 0:
            labels[i, start:] = ids[start:]
    labels[attention_mask == 0] = -100
    return labels

# row0: sys/user tokens, header, answer 'a'(=97), im_end(=99), \n(=13), then 2 pads
# row1: longer prompt, no padding
PAD = 0
ids = np.array([
    [ 5, 6, 7, 1001, 1002, 1003, 97, 99, 13, PAD, PAD],
    [ 5, 6, 7, 8, 9, 1001, 1002, 1003, 98, 99, 13],
])
am = np.array([
    [1,1,1,1,1,1,1,1,1,0,0],
    [1,1,1,1,1,1,1,1,1,1,1],
])
lab = build_labels_np(ids, am)
assert list(lab[0]) == [-100]*6 + [97, 99, 13] + [-100, -100], lab[0]
assert list(lab[1]) == [-100]*8 + [98, 99, 13], lab[1]
ok("정답 글자+종료토큰만 학습, 프롬프트/이미지/패딩은 모두 -100")

# a decoy "assistant header" inside the user text must not win (we scan from the end)
ids2 = np.array([[1001,1002,1003, 50, 51, 1001,1002,1003, 97, 99]])
am2 = np.ones_like(ids2)
lab2 = build_labels_np(ids2, am2)
assert list(lab2[0]) == [-100]*8 + [97, 99], lab2[0]
ok("헤더가 여러 번 나와도 '마지막' 것을 기준으로 삼음")

# baseline's behaviour, for contrast
baseline_labels = ids.copy()
print(f"  참고: baseline은 {int((baseline_labels != -100).sum())}개 토큰 전부 학습, "
      f"v4는 {int((lab != -100).sum())}개만 학습")

# ───────────────────────────────────────────────────────────
# 3. response_to_letter (dev pseudo-labels)
# ───────────────────────────────────────────────────────────
print("\n[3] dev 교육생 응답 -> a~d 변환")
def _norm(s): return re.sub(r"[\s\W_]+", "", str(s)).lower()
def response_to_letter(resp, options):
    if resp is None or (isinstance(resp, float) and np.isnan(resp)): return None
    s = str(resp).strip()
    if not s: return None
    low = s.lower()
    if low in LETTERS: return low
    m = re.match(r"^\(?([abcd])[\)\.\,:\s]", low)
    if m: return m.group(1)
    if low in ("1","2","3","4"): return LETTERS[int(low)-1]
    ns = _norm(s); nopts = [_norm(o) for o in options]
    if ns in nopts: return LETTERS[nopts.index(ns)]
    best, score = None, 0.0
    for i, no in enumerate(nopts):
        r = difflib.SequenceMatcher(None, ns, no).ratio()
        if r > score: best, score = i, r
    return LETTERS[best] if score >= 0.85 else None

OPTS = ["직화낙지와 오봉보쌈", "된장찌개와 불고기", "김치찌개와 삼겹살", "비빔밥과 김밥"]
cases = [
    ("c", "c"), ("C", "c"), (" b ", "b"),
    ("a) 직화낙지와 오봉보쌈", "a"), ("(d). 비빔밥과 김밥", "d"), ("b. 된장찌개", "b"),
    ("3", "c"), ("1", "a"),
    ("김치찌개와 삼겹살", "c"), ("비빔밥과 김밥 ", "d"),
    ("직화낙지와  오봉보쌈", "a"),
    ("", None), (None, None), (float("nan"), None),
    ("잘 모르겠어요", None), ("전혀 다른 문장입니다 정말로", None),
]
for inp, want in cases:
    got = response_to_letter(inp, OPTS)
    assert got == want, f"{inp!r} -> {got!r}, expected {want!r}"
ok(f"{len(cases)}개 케이스 통과 (글자/번호/본문/오타/무응답/판독불가)")

# majority vote gate
def vote(votes, min_agree=4):
    votes = [v for v in votes if v]
    if not votes: return None
    letter, cnt = Counter(votes).most_common(1)[0]
    return letter if (cnt >= min_agree and cnt/len(votes) >= 0.8) else None
assert vote(["a","a","a","a","b"]) == "a"
assert vote(["a","a","a","b","b"]) is None      # 3:2 -> 버림
assert vote(["c","c","c","c","c"]) == "c"
assert vote(["a","a","a","a",None]) == "a"      # 4/4 유효응답
assert vote(["a","a","a",None,None]) is None    # 3명뿐 -> 기준 미달
assert vote([None]*5) is None
ok("다수결 게이트: 4/5 이상 합의만 채택, 3:2는 폐기")

# ───────────────────────────────────────────────────────────
# 4. prior calibration
# ───────────────────────────────────────────────────────────
print("\n[4] 정답 분포 사전확률 보정")
PRIOR = np.array([0.40, 0.25, 0.20, 0.15]); PRIOR /= PRIOR.sum()
LOG_PRIOR = np.log(np.clip(PRIOR, 1e-6, None))
def apply_prior(p, alpha):
    if alpha == 0.0: return p
    s = np.log(np.clip(p, 1e-9, None)) + alpha * LOG_PRIOR
    s = s - s.max(axis=1, keepdims=True); e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)

P = np.array([[0.30, 0.28, 0.22, 0.20], [0.10, 0.60, 0.20, 0.10]])
for a in [0.0, 0.3, 1.0]:
    out = apply_prior(P, a)
    assert np.allclose(out.sum(1), 1.0), out
assert apply_prior(P, 0.0) is P
ok("확률 정규화 유지, alpha=0이면 완전 무변경(손해 없음)")
assert apply_prior(P, 1.0)[1].argmax() == 1
ok("확신이 강한 문항(0.60)은 사전확률로 뒤집히지 않음")

def accuracy(p, gold): return float(np.mean([LETTERS[i]==g for i,g in zip(p.argmax(1), gold)]))
def tune_alpha(p, gold):
    best_a, best_acc = 0.0, accuracy(p, gold)
    for a in [0.15,0.3,0.5,0.75,1.0]:
        acc = accuracy(apply_prior(p, a), gold)
        if acc > best_acc + 1e-9: best_a, best_acc = a, acc
    return best_a, best_acc
# prior is actively harmful here -> tuner must return alpha=0
rng = np.random.default_rng(0)
Pn = rng.dirichlet([2,2,2,2], size=200)
gold_bad = [LETTERS[i] for i in Pn.argmax(1)]
a, acc = tune_alpha(Pn, gold_bad)
assert a == 0.0 and acc == 1.0, (a, acc)
ok("보정이 해로울 땐 alpha=0 자동 선택")

# ───────────────────────────────────────────────────────────
# 5. candidate selection with margin
# ───────────────────────────────────────────────────────────
print("\n[5] 최종 후보 선택 (마진 규칙)")
def select(res, margin=0.005):
    best = "zeroshot"
    for name in ["finetuned", "ensemble"]:
        if name in res and res[name][1] > res[best][1] + margin:
            best = name
    return best
assert select({"zeroshot":(0,.700), "finetuned":(0,.703)}) == "zeroshot"   # noise
assert select({"zeroshot":(0,.700), "finetuned":(0,.740)}) == "finetuned"
assert select({"zeroshot":(0,.700), "finetuned":(0,.650)}) == "zeroshot"   # overfit -> discard
assert select({"zeroshot":(0,.700), "finetuned":(0,.740), "ensemble":(0,.760)}) == "ensemble"
assert select({"zeroshot":(0,.700)}) == "zeroshot"                          # finetune off
ok("박빙이면 단순한 쪽, 확실히 이길 때만 교체, 파인튜닝 실패 시 자동 폐기")

# ───────────────────────────────────────────────────────────
# 6. train-time choice shuffle keeps gold consistent
# ───────────────────────────────────────────────────────────
print("\n[6] 학습용 보기 셔플 증강")
for i in range(500):
    opts = ["W","X","Y","Z"]; gold = i % 4
    rng2 = random.Random(42*1_000_003 + i)
    perm = list(range(4)); rng2.shuffle(perm)
    shown = [opts[perm[k]] for k in range(4)]
    new_gold = perm.index(gold)
    assert shown[new_gold] == opts[gold]
ok("셔플 후에도 정답 글자가 항상 정답 '내용'을 가리킴 (500회)")
seen = set()
for e in range(4):
    rng3 = random.Random(42*1_000_003 + e*7919 + 0)
    p = list(range(4)); rng3.shuffle(p); seen.add(tuple(p))
ok(f"에폭마다 다른 순열 생성: {len(seen)}/4 가지")

print("\n" + "="*50)
print("모든 로직 테스트 통과 ✅")
