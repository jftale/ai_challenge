# -*- coding: utf-8 -*-
"""MIGRATION.md의 '바꾸기 전' 코드가 실제 베이스라인 v3와 일치하는지 검사한다.

가이드를 고친 뒤 아래를 실행하세요:

    python tests/verify_migration_guide.py

원본에 없는 코드를 "바꾸기 전"이라고 적어두면, 그대로 따라 한 사람이 에러를 만납니다.
불일치가 있으면 종료 코드 1로 끝납니다.
"""
import re, io, os, sys, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
base = io.open(os.path.join(HERE, "baseline_v3_reference.py"), encoding="utf-8").read()
base_lines = [x.strip() for x in base.split("\n") if x.strip()]
base_norm = re.sub(r"[ \t]+", " ", base)
md = io.open(os.path.join(HERE, os.pardir, "MIGRATION.md"), encoding="utf-8").read()

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
    sys.exit(1)
else:
    print("✅ 모든 '바꾸기 전' 블록이 원본과 일치")
