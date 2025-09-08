# %% installs
!pip install konlpy jpype1
!pip install wordcloud
!pip install matplotlib
!pip install -U seaborn
# %% imports
import os
import pandas as pd
import seaborn as sns
import re, collections
import torch
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
# %%
folder = 'data/comments'
files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.csv')]

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)
    
all_comments_df = pd.concat(dfs, ignore_index=True)
# %%
print(all_comments_df.info())
print(all_comments_df.head())

# %%
print(all_comments_df[all_comments_df.isnull().any(axis=1)])
all_comments_df = all_comments_df.dropna()
print(all_comments_df.info())
print(all_comments_df.head())
# %% merge

all_comments_labeled = all_comments_df.merge(
    pd.read_csv("data/meta/hwi_videos_labeled.csv", usecols=["video_id","label_rule"]).drop_duplicates("video_id"),
    on="video_id", how="left"
)

all_comments_labeled.to_csv("data/meta/all_comments_labeled.csv", index=False, encoding="utf-8-sig")

# %% A) 댓글 반응 긍정/중립/부정 라벨링

BASE_LEXICON = set(["좋다","최고","대박","별로","최악","짜증","화남"])
SLANG_RE = re.compile(r"(ㅋㅋ+|ㅎㅎ+|ㅠ+|ㅜ+|ㄷㄷ+|ㄹㅇ|ㅇㅈ|ㄱㄱ|노잼|핵.*|미쳤|레전드|개\w+)")
POS_HINTS = ["ㅋㅋ","ㅎㅎ","😊","😍","👍","레전드","지리","ㄹㅇ","존버성공"]
NEG_HINTS = ["ㅠㅠ","ㅜㅜ","😡","🤮","👎","노잼","혐","빡침","극혐"]

POS_DICT = {"레전드","지리네","갓","꿀","혜자","개이득","사랑해","쵝오","최고","대박","귀여워"}
NEG_DICT = {"노잼","극혐","구림","빡침","붕괴","망했","사기","혐오","역겹","최악","별로","짜증","실망"}

LAUGH = re.compile(r"(ㅋㅋ+|ㅎㅎ+)")
NEGATOR = re.compile(r"(안|못|별로|아닌)")


def extract_slang_candidates(texts, topn=200, min_freq=5):
    freq = collections.Counter()
    for t in texts:
        for tok in re.findall(r"[가-힣A-Za-z0-9]+|[ㅋㅎㅠㅜㄷ]+", str(t)):
            if tok not in BASE_LEXICON and (SLANG_RE.search(tok) or len(tok) <= 4):
                freq[tok] += 1
    return [w for w, c in freq.most_common(topn) if c >= min_freq]


def weak_polarity_score(texts, candidates):
    co_pos, co_neg, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    for t in texts:
        t = str(t)
        present = {w for w in candidates if w in t}
        pos_hit = any(h in t for h in POS_HINTS)
        neg_hit = any(h in t for h in NEG_HINTS)
        for w in present:
            cnt[w] += 1
            if pos_hit: co_pos[w] += 1
            if neg_hit: co_neg[w] += 1
    scores = {}
    for w in candidates:
        p = (co_pos[w] + 1) / (cnt[w] + 2)
        n = (co_neg[w] + 1) / (cnt[w] + 2)
        scores[w] = p - n
    return scores


def classify_sentiment(text):
    t = str(text)
    pos = any(w in t for w in POS_DICT) or bool(LAUGH.search(t))
    neg = any(w in t for w in NEG_DICT)
    if pos and NEGATOR.search(t):  # "안/못/별로/아닌" 등 부정어가 긍정표현과 함께 나오면 뒤집기
        pos, neg = False, True
    if pos and not neg: return "긍정"
    if neg and not pos: return "부정"
    return "중립"



CANDIDATE_MODELS = [
    ("tabularisai/multilingual-sentiment-analysis", "3class"),  # negative/neutral/positive
    ("daekeun-ml/koelectra-small-v3-nsmc", "binary"),          # neg/pos
    ("sangrimlee/bert-base-multilingual-cased-nsmc", "binary") # neg/pos
]

def load_sentiment_pipeline():
    last_err = None
    for mid, mtype in CANDIDATE_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(mid)
            mdl = AutoModelForSequenceClassification.from_pretrained(mid)
            pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, return_all_scores=True)
            return pipe, mtype
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"모델 로드 실패: {last_err}")

pipe, MODEL_TYPE = load_sentiment_pipeline()
LABEL_MAP_3 = {"negative":"부정", "neutral":"중립", "positive":"긍정"}

def refine_with_plm(text, neutral_band=0.15):
    scores = pipe(str(text))[0]  # [{'label': 'positive', 'score': 0.7}, ...]
    if MODEL_TYPE == "3class":
        label = max(scores, key=lambda x: x["score"])["label"].lower()
        return LABEL_MAP_3.get(label, "중립")
    
    pos = next((s["score"] for s in scores if "pos" in s["label"].lower()), None)
    neg = next((s["score"] for s in scores if "neg" in s["label"].lower()), None)
    if pos is None or neg is None:
        label = max(scores, key=lambda x: x["score"])["label"].lower()
        return "긍정" if "pos" in label else "부정"
    if abs(pos - neg) < neutral_band:
        return "중립"
    return "긍정" if pos > neg else "부정"


def classify_hybrid(text):
    rule = classify_sentiment(text)
    if rule == "중립":
        return refine_with_plm(text)
    return rule

# %%


slang_cands = extract_slang_candidates(all_comments_labeled["text"], topn=200, min_freq=5)
weak_scores = weak_polarity_score(all_comments_labeled["text"], slang_cands)


all_comments_labeled["final_sentiment"] = all_comments_labeled["text"].map(classify_hybrid)

# 저장
all_comments_labeled.to_csv("data/meta/all_comments_final_sentiment.csv", index=False, encoding="utf-8-sig")
# %%
data = pd.read_csv("data/meta/all_comments_final_sentiment.csv")
print(data["final_sentiment"].value_counts())
# %%
print(data[data["final_sentiment"] == "긍정"]['text'])

## 애초에 대부분 긍정이긴함 감정 분석은 어려운듯


# %% B) 영상별 댓글 키워드
INPUT_CSV = "data/meta/all_comments_final_sentiment.csv"
VIDEO_ID_COL = "video_id"
COMMENT_COL = "text"
LABEL_COL = "label_rule"
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"
TOP_K = 5

okt = Okt()

STOPWORDS_KO = {
    "있다","아니다","하다","되다","없다","그렇다","그리고","하지만","너무","진짜","정말",
    "이거","저거","그거","요거","거기","여기","저기"
}

def tokenize_ko(text: str):
    text = re.sub(r"[^가-힣0-9a-zA-Z\s]", " ", str(text))
    return [w for w, p in okt.pos(text, stem=True, norm=True)
            if p in ("Noun", "Adjective") and len(w) >= 2 and w not in STOPWORDS_KO]


df = pd.read_csv(INPUT_CSV)

# 댓글 컬럼/비디오ID 컬럼 존재 확인
assert VIDEO_ID_COL in df.columns, f"{VIDEO_ID_COL} 컬럼이 없음"
assert COMMENT_COL  in df.columns, f"{COMMENT_COL} 컬럼이 없음"

docs = (df.groupby(VIDEO_ID_COL)[COMMENT_COL]
          .apply(lambda x: " ".join(map(str, x)))
          .reset_index()
          .rename(columns={COMMENT_COL: "doc_text"}))


vectorizer = TfidfVectorizer(tokenizer=tokenize_ko, max_features=5000)
X = vectorizer.fit_transform(docs["doc_text"])
feature_names = vectorizer.get_feature_names_out()


top_keywords = []
X_csr = X.tocsr()
for i in range(X_csr.shape[0]):
    row = X_csr.getrow(i)
    if row.nnz == 0:
        top_keywords.append([])
        continue
    idx = row.indices[row.data.argsort()[::-1][:TOP_K]]
    kws = [feature_names[j] for j in idx]
    top_keywords.append(kws)

docs["top_keywords"] = top_keywords
print(docs[[VIDEO_ID_COL, "top_keywords"]].head(10))


if LABEL_COL is not None and LABEL_COL in df.columns:
    video_labels = df[[VIDEO_ID_COL, LABEL_COL]].drop_duplicates(VIDEO_ID_COL)
    docs = docs.merge(video_labels, on=VIDEO_ID_COL, how="left")

    label_keywords = {}
    for label, g in docs.groupby(LABEL_COL):
        c = Counter()
        for kws in g["top_keywords"]:
            c.update(kws)
        label_keywords[label] = [w for w, _ in c.most_common(5)]

    print("\n라벨별 Top5 키워드")
    for label, words in label_keywords.items():
        print(f"- {label}: {', '.join(words)}")
# %% 영상별 댓글 좋아요 합, 댓글 합 시각화
from textwrap import shorten

df = pd.read_csv(INPUT_CSV)
df_t = pd.read_csv("data/meta/hwi_videos_labeled.csv", usecols=["video_id", "title"])
# 윈도우 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지


for col in ["video_id", "comment_id", "like_count"]:
    assert col in df.columns, f"{col} 컬럼이 없음"
    
df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

agg = (df.groupby("video_id")
       .agg(
           comment = ("comment_id", "nunique"),
           like = ("like_count", "sum"),
           first_published=("published_at", "min"),
       ).reset_index()
       )
agg["like_per_comment"] = agg["like"] / agg["comment"].fillna(0)

agg = agg.merge(df_t.drop_duplicates("video_id"), on="video_id", how="left")
# 너무 긴 제목은 잘라서 표시 (예: 30자)
agg["title_short"] = agg["title"].fillna("").apply(lambda s: shorten(str(s), width=30, placeholder="…"))
topn = 20

print(agg.head())
## 시각화 댓글 수 TopN
top_comments = agg.sort_values("comment", ascending=False).head(topn)
plt.figure(figsize=(12,6))
plt.bar(top_comments["title_short"].astype(str), top_comments["comment"])
plt.xticks(rotation=60, ha='right')
plt.ylabel("댓글 수")
plt.title(f"영상별 댓글 수 Top {topn}")
plt.tight_layout()
plt.show()

# 댓글 좋아요 합계 TopN

top_likes = agg.sort_values("like", ascending=False).head(topn)
plt.figure(figsize=(12,6))
plt.bar(top_likes["title_short"].astype(str), top_likes["like"], color='orange')
plt.xticks(rotation=60, ha='right')
plt.ylabel("댓글 좋아요 수")
plt.title(f"영상별 댓글 좋아요 수 Top {topn}")
plt.tight_layout()
plt.show()

# 산점도
plt.figure(figsize=(6,5))
plt.scatter(agg["comment"], agg["like"])
plt.xlabel("댓글 수")
plt.ylabel("댓글 좋아요 합계")
plt.title("댓글 수 vs 댓글 좋아요 합계 (영상별)")
plt.tight_layout()
plt.show()

# 댓글당 좋아요 TopN (댓글 5개 이상)

top_lpc = (agg[agg["comment"] >= 5]
           .sort_values("like_per_comment", ascending=False)
           .head(topn))
plt.figure(figsize=(12,6))
plt.bar(top_lpc["title_short"].astype(str), top_lpc["like_per_comment"])
plt.xticks(rotation=60, ha="right")
plt.ylabel("like per comment")
plt.title(f"댓글당 좋아요 Top{topn} (댓글≥5)")
plt.tight_layout()
plt.show()
# %%
# 3) 좋아요 많은 순으로 정렬 후 상위 5개 추출
top5_comments = df.sort_values("like_count", ascending=False).head(5)

# 4) 필요한 컬럼만 보기 좋게 출력
print(top5_comments[["comment_id","video_id","author","text","like_count"]])



# %% 라벨별 댓글 좋아요 합, 댓글 합 시각화
agg_label = (df.groupby("label_rule")
       .agg(
           comment = ("comment_id", "nunique"),
           like = ("like_count", "sum"),
           video_count=("video_id","nunique"),
       ).reset_index()
       )
# 평균 댓글 수, 평균 좋아요 계산
agg_label["comment_mean"] = agg_label["comment"] / agg_label["video_count"]
agg_label["like_mean"] = agg_label["like"] / agg_label["video_count"]

# 라벨별 댓글 수 평균
plt.figure(figsize=(12,6))
plt.bar(agg_label["label_rule"], agg_label["comment_mean"])
plt.xticks(rotation=60, ha='right')
plt.ylabel("댓글 수")
plt.title("라벨별 평균 댓글 수")
plt.tight_layout()
plt.show()

# 댓글 좋아요 평균

plt.figure(figsize=(12,6))
plt.bar(agg_label["label_rule"], agg_label["like_mean"], color='orange')
plt.xticks(rotation=60, ha='right')
plt.ylabel("댓글 좋아요 수")
plt.title("라벨별 평균 댓글 좋아요 수")
plt.tight_layout()
plt.show()



# %%
print(df['published_at'])
# %% 시계열 관련 분석
# 1) 댓글이 몰리는 시간대 분석
# 2) 댓글이 몰리는 요일 분석
df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True)

# 한국 시간대(KST)로 맞추기, +00:00 은 UTC 오프셋을 의미
df["published_at"] = df["published_at"].dt.tz_convert("Asia/Seoul")

df["hour"] = df["published_at"].dt.hour
df["weekday"] = df["published_at"].dt.day_name()

# %%
hourly = df.groupby("hour")["comment_id"].count().reset_index()

pivot = df.pivot_table(index="weekday", columns="hour", values="comment_id", aggfunc="count", fill_value=0)

# 시간대별 막대그래프
plt.figure(figsize=(10,5))
plt.bar(hourly["hour"], hourly["comment_id"])
plt.xticks(range(0,24))
plt.xlabel("시간대 (시)")
plt.ylabel("댓글 수")
plt.title("시간대별 댓글 수 분포")
plt.show()


# 요일-시간대 히트맵
plt.figure(figsize=(12,6))
sns.heatmap(pivot, cmap="YlOrRd")
plt.title("요일/시간대별 댓글 수 Heatmap")
plt.xlabel("시간대 (시)")
plt.ylabel("요일")
plt.show()

# %% 단순 합은 특정 업로드 요일에 몰릴수 있음, 평균으로 계산

df["weekday_num"] = df["published_at"].dt.dayofweek  # 0=월 ~ 6=일
df["weekday_name"] = df["published_at"].dt.day_name()

# 각 영상(video_id)별 댓글 수
video_comments = df.groupby(["video_id","weekday_num","weekday_name"])["comment_id"].count().reset_index()

# 요일별 평균 (영상당 댓글 수)
weekday_avg = video_comments.groupby(["weekday_num","weekday_name"])["comment_id"].mean().reset_index()
weekday_avg = weekday_avg.sort_values("weekday_num")

# 시각화
plt.figure(figsize=(10,5))
plt.bar(weekday_avg["weekday_name"], weekday_avg["comment_id"])
plt.xlabel("요일")
plt.ylabel("평균 댓글 수 (영상당)")
plt.title("요일별 평균 댓글 수 분포")
plt.show()
# %%
