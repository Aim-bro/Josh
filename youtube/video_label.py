# %% import 정리
import pandas as pd
import re
# %% 
df = pd.read_csv('data/meta/hwi_videos_list.csv')

# 패턴 정의
emo = re.compile(r"(헉|헐|와|ㄷㄷ|대박|충격|미쳤|최고|지리|레전드|부러워|화나|무서워|슬퍼|행복|후회|어떻게|믿기|너무|진짜)")
lol = re.compile(r"(ㅋㅋ+|ㅎㅎ+|lol|lmao|아니죠|아니잖)")
qmark = re.compile(r"\?$")
story = re.compile(r"(사기|사건|방송사고|신제품|리뷰|비교|분석|실험|해봤|챌린지|브이로그|먹방|여행|룩|튜토리얼|가이드|정리|요약|업데이트|버그|공지)")

# %% 함수 정의
def simple_feats(title: str):
    t = str(title).strip()
    tokens = re.findall(r"[가-힣A-Za-z0-9]+", t)
    return {
        "len_tokens": len(tokens),
        "has_ellipsis": ("..." in t) or (".." in t) or ("…") in t,
        "ends_q": t.endswith("?") or bool(qmark.search(t)),
    }

def classify_title(title: str) -> str:
    t = str(title).strip()
    if lol.search(t) or qmark.search(t):
        return "밈/유머"
    if emo.search(t):
        return "감정/반응"
    if story.search(t):
        return "상황/스토리"
    f = simple_feats(t)
    if f["len_tokens"] <= 3 and (f["has_ellipsis"] or f["ends_q"]):
        return "밈/유머"
    return "상황/스토리"

# %%
df["label_rule"] = df["title"].astype(str).map(classify_title)


path = r"data/meta/hwi_videos_labeled.csv"
df.to_csv(path, index=False, encoding="utf-8-sig")

# 라벨 분포 확인
print(df["label_rule"].value_counts())
print("저장 완료:", path)