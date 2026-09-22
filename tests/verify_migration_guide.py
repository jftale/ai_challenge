# -*- coding: utf-8 -*-
import re, io, difflib

base = io.open("baseline_v3.py", encoding="utf-8").read()
base_lines = [x.strip() for x in base.split("\n") if x.strip()]
base_norm = re.sub(r"[ \t]+", " ", base)
md = io.open("/home/user/ai_challenge/MIGRATION.md", encoding="utf-8").read()

# 1) "### 바꾸기 전" 헤딩 뒤의 첫 python 블록
#  2) "# 바꾸기 전" 주석으로 시작하는 블록
cands = []
for m in re.finditer(r"###\s*바꾸기 전\s*\n+```python\n(.*?)```", md, re.S):
    cands.append(("헤딩", m.start(), m.group(1)))
for m in re.finditer(r"```python\n(# 바꾸기 전.*?)```", md, re.S):
    cands.append(("주석", m.start(), m.group(1)))
cands.sort(key=lambda x: x[1])

# 문서 내 위치 -> 어느 단계인지
steps = [(m.start(), m.group(1)) for m in re.finditer(r"^## (\d+단계\..*)$", md, re.M)]
def step_of(pos):
    cur = "(단계 밖)"
    for p, t in steps:
        if p < pos: cur = t
    return cur

print(f"'바꾸기 전' 블록 {len(cands)}개 검사\n" + "=" * 74)
problems = []
for n, (kind, pos, b) in enumerate(cands, 1):
    lines = [l for l in b.split("\n")
             if l.strip() and not l.strip().startswith("#") and l.strip() != "..."]
    missing = []
    for l in lines:
        core = re.sub(r"\s+#.*$", "", l).strip()
        if not core or core.startswith("!pip"):     # pip 셀은 참조파일에 없음(정상)
            continue
        if re.sub(r"[ \t]+", " ", core) not in base_norm:
            missing.append(core)
    tag = "✅" if not missing else "❌"
    print(f"[{n}] {tag} {step_of(pos)}  ({kind})")
    print(f"     첫 줄: {(lines[0].strip() if lines else '(없음)')[:66]}")
    for x in missing:
        print(f"     └─ 원본에 없음: {x[:68]}")
        c = difflib.get_close_matches(x, base_lines, n=1, cutoff=0.55)
        if c: print(f"          원본은: {c[0][:68]}")
    if missing: problems.append((step_of(pos), missing))
    print()

print("=" * 74)
if problems:
    print(f"❌ 수정 필요: {len(problems)}곳")
    for s, _ in problems: print("   -", s)
else:
    print("✅ 모든 '바꾸기 전' 블록이 원본과 일치")
