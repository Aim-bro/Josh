# %%
print("let's go")
# %%
# !pip install -q -U sentence-transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('jhgan/ko-sroberta-multitask')


# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.cluster import KMeans


# %%
date = '20231122'
page = 1
url = f'https://finance.naver.com/news/mainnews.naver?date={date}&page={page}'

# HTTP GET 요청을 보내고 HTML 페이지 가져오기
response = requests.get(url)
html_content = response.text

soup = BeautifulSoup(html_content, 'html.parser')

# td 태그에서 페이지 번호 가져오기
td_tag = soup.find('td', class_='pgRR')
if td_tag:
    link_tag = td_tag.find('a')
    if link_tag:
        href = link_tag['href']
        # href에서 page= 다음에 오는 페이지 번호 추출
        page_number = href.split('page=')[-1]
        print("페이지 번호:", page_number)

# %% 페이지 끝까지
from datetime import datetime, timedelta
# date = '20231122'
article_titles = []  # 기사 제목을 저장할 리스트
date_start = '20231101'
date_end = '20231123'

# 문자열 형식의 날짜를 datetime 형식으로 변환
start_date = datetime.strptime(date_start, '%Y%m%d')
end_date = datetime.strptime(date_end, '%Y%m%d')

current_date = start_date
while start_date <= end_date:

    page_number = 1
    current_date = start_date.strftime('%Y%m%d')
    url = f'https://finance.naver.com/news/mainnews.naver?date={current_date}&page={page}'

    # HTTP GET 요청을 보내고 HTML 페이지 가져오기
    response = requests.get(url)
    html_content = response.text
    
    soup = BeautifulSoup(html_content, 'html.parser')

    # td 태그에서 페이지 번호 가져오기
    td_tag = soup.find('td', class_='pgRR')
    if td_tag:
        link_tag = td_tag.find('a')
        if link_tag:
            href = link_tag['href']
            # href에서 page= 다음에 오는 페이지 번호 추출
            page_number = href.split('page=')[-1]
            print("페이지 번호:", page_number)

    for i in range(1,int(page_number)):
        url = f'https://finance.naver.com/news/mainnews.naver?date={current_date}&page={i}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        titles = soup.find_all('dd', class_='articleSubject')
        
        for title in titles:
            article_title = title.text.strip()
            article_titles.append(article_title)

    start_date += timedelta(days=1)  # 날짜를 하루씩 증가시킴    

df = pd.DataFrame({'titles': article_titles})
# %%
df
# %%
df['titles'] = df['titles'].replace(r'[^가-힣 ]', ' ', regex=True)\
.replace("'", '').replace(r'\s+', ' ', regex=True).str.strip().str[:255]

df.head()
# %%
df = df[df['titles'].str.strip().astype(bool)]

len(df)
# %%
df['titles'].values.tolist()[:5]
# %%
corpus = df['titles'].values.tolist()

embeddings = model.encode(corpus)

embeddings[:5]
# %%
num_clusters = 4
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(corpus[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print('Cluster %d (%d)' % (i+1, len(cluster)))
    print(cluster)
    print('')



# %%
import os
from konlpy.tag import Komoran, Okt, Kkma, Hannanum
from tqdm import tqdm

extractor = Hannanum()

nouns = [] # 명사만 추출해라

for review in tqdm(df['titles'].values.tolist()):
    nouns.extend(extractor.nouns(review))

len(nouns)
# %%
#   jvmpath = jvmpath or jpype.getDefaultJVMPath()

from collections import Counter

count = Counter(nouns)
words = dict(count.most_common())

for i, (word, count) in enumerate(words.items()):
    if i > 10:
        break

    print(word, count)
# %%

import requests

url = 'https://github.com/kairess/MBTI-wordcloud/raw/master/NanumSquareRoundR.ttf'
response = requests.get(url)

# 응답 상태코드 확인
if response.status_code == 200:
    with open('NanumSquareRoundR.ttf', 'wb') as file:
        file.write(response.content)
    print('파일 다운로드 및 저장 완료')
else:
    print('파일을 다운로드할 수 없습니다.')


# %%

from wordcloud import WordCloud
import matplotlib.pyplot as plt

wc = WordCloud(
    font_path='NanumSquareRoundR.ttf',
    width=2000,
    height=1000
).generate_from_frequencies(words)

plt.figure(figsize=(20, 10))
plt.imshow(wc)
plt.axis('off')
plt.show()

# %%
