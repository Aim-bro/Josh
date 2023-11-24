# %%
print("let's go")
# %%
# !pip install -q -U sentence-transformers
from sentence_transformers import SentenceTransformer

# %%
model = SentenceTransformer('jhgan/ko-sroberta-multitask')
sentences = ['안녕하세요?', '한국어 문장 임베딩을 위한 버트 모델이다.']
embeddings = model.encode(sentences)

print(embeddings)
# %% clustering 예제

from sklearn.cluster import KMeans

# Corpus with example sentences
sentences = ['한 남자가 음식을 먹는다.',
          '한 남자가 빵 한 조각을 먹는다.',
          '그 여자가 아이를 돌본다.',
          '한 남자가 말을 탄다.',
          '한 여자가 바이올린을 연주한다.',
          '두 남자가 수레를 숲 속으로 밀었다.',
          '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
          '원숭이 한 마리가 드럼을 연주한다.',
          '치타 한 마리가 먹이 뒤에서 달리고 있다.',
          '한 남자가 파스타를 먹는다.',
          '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
          '치타가 들판을 가로 질러 먹이를 쫓는다.']

embeddings = model.encode(sentences)

# Then, we perform k-means clustering using sklearn:
num_clusters = 5
clustering_model = KMeans(n_clusters=num_clusters)
clustering_model.fit(embeddings)
cluster_assignment = clustering_model.labels_

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(sentences[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")

# %%
import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL 설정 (특정 날짜와 페이지 번호 입력)
date = '20231122'
page = 1
url = f'https://finance.naver.com/news/mainnews.naver?date={date}&page={page}'

# HTTP GET 요청을 보내고 HTML 페이지 가져오기
response = requests.get(url)
html_content = response.text

# BeautifulSoup을 사용하여 HTML 파싱
soup = BeautifulSoup(html_content, 'html.parser')
# 기사 제목을 담고 있는 요소 찾기
titles = soup.find_all('dd', class_='articleSubject')

# 각 기사 제목 출력
for title in titles:
    article_title = title.find('a').text.strip()
    print(article_title)



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

date = '20231122'
page_number = 1
url = f'https://finance.naver.com/news/mainnews.naver?date={date}&page={page}'

# HTTP GET 요청을 보내고 HTML 페이지 가져오기
response = requests.get(url)
html_content = response.text
article_titles = []  # 기사 제목을 저장할 리스트
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
    url = f'https://finance.naver.com/news/mainnews.naver?date={date}&page={i}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    titles = soup.find_all('dd', class_='articleSubject')
    
    for title in titles:
        article_title = title.find('a').text.strip()
        article_titles.append(article_title)
    

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
# !pip install -q konlpy tqdm
# konlpy를 사용하려면
# Java와 JPype가 필요합니다. Java가 설치되어 있지 않다면 Java를 설치해야 합니다. 
# 그리고 JPype는 Python에서 Java와의 상호작용을 도와주는 라이브러리입니다.
# 이러한 사전 요구사항들을 충족시켜야 konlpy를 올바르게 사용할 수 있습니다.
import os
# Replace 'path_to_java' with the actual path where Java is installed
# os.environ['JAVA_HOME'] = 'path_to_java'
# 껐다키니까 되네
# %%
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
    # 파일 저장
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
