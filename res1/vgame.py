# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns


# %%
df = pd.read_csv('vgames2.csv')
print(df.shape)
df.head()
# %%
df.isnull().sum()
# null값 처리 순서대로 year부터 genre, publisher
# %%
df[df.Year.isnull()]
# 연도를 알 수 있는 힌트가 존재하지 않음
# 모두 입력하기 보단 다 drop시키거나 특정 값 이상의 경우만 찾아보는것은 어떨까
# %%
df[df.Genre.isnull()]
# 장르는 회사와 이름이 비슷한 시리즈 게임으로 유추할 수 있지 않을까?
# %%
df[df.Publisher.isnull()]
# 몇 개 빼곤 감이 안온다
# %%
df[df.duplicated()]
# 중복되는 값은 없다
# %%
df_a = df.copy()
# 원본 유지
# %%
df_a = df_a.drop(df.iloc[:,0].name, axis=1)
# 필요 없어보이는 행 삭제
# %%
print(df_a.shape)
# %% 데이터를 살펴보니 뭔가 이상하다 M백만 단위는 소수점 첫자리 K천 단위는 수백까지도 나와 편차가 크다
# 아무래도 모두 소수점대 M가 맞고 K는 1000으로 나누는게 맞는것 같다
def convert_to_number(value):
    if 'K' in value:
        return float(value.replace('K', '')) / 1000
    elif 'M' in value:
        return float(value.replace('M', ''))
    else:
        return float(value)

df_a['NA_Sales'] = df_a['NA_Sales'].apply(convert_to_number)
df_a['EU_Sales'] = df_a['EU_Sales'].apply(convert_to_number)
df_a['JP_Sales'] = df_a['JP_Sales'].apply(convert_to_number)
df_a['Other_Sales'] = df_a['Other_Sales'].apply(convert_to_number)

df_a['Total_Sales'] = df_a.loc[:, 'NA_Sales':'Other_Sales'].sum(axis=1)
# %%
# 총 매출을 보기 위해
# df_a.sort_values(by='Total_Sales',ascending=False).head()

# 연도가 null값인 것 중에 판매량이 상당한건 찾아볼 가치가 있지 않을까?
print(df_a.Total_Sales.describe())
# df_a[df_a['Year'].isnull()].sort_values(by='Total_Sales',ascending=False).head(10)

# 평균 판매량 0.53, 중위값 0.17
#
# 상위값들 보기 위해
# df_a[df_a['Year'].isnull() & (df_a['Total_Sales'] > 0.47)].sort_values(by='Total_Sales', ascending=False)

# %%
df_a.loc[df_a['Name'] == 'The Golden Compass', 'Year'] = 2007
df_a.loc[df_a['Name'] == 'Pac-Man Fever', 'Year'] = 2002
df_a.loc[df_a['Name'] == 'Shrek the Third', 'Year'] = 2007
df_a.loc[df_a['Name'] == 'Madden NFL 07', 'Year'] = 2006
df_a.loc[df_a['Name'] == 'Silent Hill: Homecoming', 'Year'] = 2008
df_a.loc[df_a['Name'] == "Tom Clancy's Rainbow Six: Critical Hour", 'Year'] = 2007
df_a.loc[df_a['Name'] == 'Haven: Call of the King', 'Year'] = 2002
df_a.loc[df_a['Name'] == 'Madden NFL 11', 'Year'] = 2010
df_a.loc[df_a['Name'] == 'The Chronicles of Riddick: Escape from Butcher Bay', 'Year'] = 2004
df_a.loc[df_a['Name'] == 'Madden NFL 2004', 'Year'] = 2003
df_a.loc[df_a['Name'] == 'FIFA Soccer 2004', 'Year'] = 2010
df_a.loc[df_a['Name'] == 'wwe Smackdown vs. Raw 2006', 'Year'] = 2005
df_a.loc[df_a['Name'] == 'Space Invaders', 'Year'] = 1978
df_a.loc[df_a['Name'] == "Frogger's Adventures: Temple of the Frog", 'Year'] = 2001
df_a.loc[df_a['Name'] == 'LEGO Indiana Jones: The Original Adventures', 'Year'] = 2008
df_a.loc[df_a['Name'] == 'Call of Duty 3', 'Year'] = 2006
df_a.loc[df_a['Name'] == 'Call of Duty: Black Ops', 'Year'] = 2010
df_a.loc[df_a['Name'] == 'LEGO Harry Potter: Years 5-7', 'Year'] = 2011
df_a.loc[df_a['Name'] == 'LEGO Batman: The Videogame', 'Year'] = 2008


# %%
df_a = df_a.drop(df_a[df_a.Year.isnull()].index) #  Year 결측치 제거
# df_a.isnull().sum()

# print(np.sort(df_a.Year.unique()))
# 1980년 이전 부터 이상값이 존재
# 0~16, 86~98까지의 연도가 존재
# 2000년대, 1900년대의 게임이 아닐까 의심(정답)
#  이걸 하기 위해 먼저 nan값 처리를 먼저 해줘야함
# df_a['Year'] = df_a['Year'].astype(int)
df_a.loc[df_a['Year'] <= 20, 'Year'] += 2000
cond = (df_a.Year < 100) & (df_a.Year > 60)
df_a.loc[cond,'Year'] += 1900
# %%
# 장르
# df_a.isnull().sum()
# df_a[df_a['Genre'].isnull()].sort_values(by='Total_Sales', ascending=False)

df_a[df_a['Name'].str.contains('Pokemon') & (df_a['Platform'] == '3DS')]
df_a.loc[df_a['Name'] == 'Pokemon X/Pokemon Y', 'Genre'] = 'Role-Playing'

df_a[df_a['Name'].str.contains('Wii')]
df_a.loc[df_a['Name'] == 'Wii Party', 'Genre'] = 'Misc'

df_a[df_a['Name'].str.contains('Final Fantasy')]
df_a.loc[df_a['Name'] == 'Final Fantasy XII', 'Genre'] = 'Role-Playing'
df_a = df_a.drop(df_a[df_a.Genre.isnull()].index) #  Genre 결측치 제거
# %% publisher
df_a[df_a['Publisher'].isnull()].sort_values(by='Total_Sales', ascending=False)

df_a[df_a['Name'].str.contains('wwe')]
df_a.loc[df_a['Name'] == 'wwe Smackdown vs. Raw 2006', 'Publisher'] = 'THQ'

df_a = df_a.drop(df_a[df_a.Publisher.isnull()].index) #  Publisher 결측치 제거

# %%
df_a.describe()
# df_a.isnull().sum()

# %% 장르에 관해
# 1. 장르별 총 판매량
genre_sales = df_a.groupby('Genre')['Total_Sales'].sum().reset_index().sort_values(by='Total_Sales', ascending=False)

plt.figure(figsize=(10, 6))
plt.rc('font', family='Malgun Gothic')
sns.set(style="whitegrid")
ax = sns.barplot(x='Total_Sales', y='Genre', data=genre_sales, palette='viridis')
ax.set(xlabel='Total Sales', ylabel='Genre', title='Total Sales Genre')
ax.set_title('Total Sales Genre', pad=20)
plt.show()

# %%
# 2. 장르별 국가 판매량
sales_by_genre_country = df_a.groupby('Genre')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()

# 데이터 시각화 (lineplot)
plt.figure(figsize=(12, 8))
sns.lineplot(data=sales_by_genre_country.melt(id_vars='Genre', var_name='Country', value_name='Sales'),
             x='Genre', y='Sales', hue='Country')
plt.title('장르별 국가 판매량 추이')
plt.xlabel('장르')
plt.ylabel('총 판매량')
plt.xticks(rotation=90)
plt.legend(title='국가')
plt.show()

# %%
# 3. 장르별 플랫폼
sales_by_genre_platform = df_a.groupby(['Genre','Platform']).size()
sales_by_genre_platform = sales_by_genre_platform.reset_index(name='Counts')

action_sales = sales_by_genre_platform[sales_by_genre_platform['Genre'] == 'Action']
action_sales = action_sales.sort_values('Counts', ascending=False)

# 시각화
plt.figure(figsize=(12, 8))
sns.barplot(x='Genre', y='Counts', hue='Platform', data=action_sales)
plt.title('Action Counts')
plt.xlabel('Genre')
plt.ylabel('Counts')
plt.xticks(rotation=45)
plt.legend(title='Platform')
plt.tight_layout()
plt.show()

# %% 연도
# 1. 연도별 장르 점유율
df_a['Year_Group'] = df_a['Year'] // 10 * 10

# 장르와 연도 그룹에 따른 총 판매량 계산
sales_by_genre_year_group = df_a.groupby(['Genre', 'Year_Group']).agg({'Total_Sales': 'sum'}).reset_index()

# 시각화 (선 그래프)
plt.figure(figsize=(12, 8))
sns.lineplot(data=sales_by_genre_year_group, x='Year_Group', y='Total_Sales', hue='Genre')
plt.title('Per 10-Year Genre Total Sales')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.show()

# %% EDA한 df csv 저장
# df_a.to_csv('new_vgames.csv', index=False)





# %%
max_sales_idx = df_a.groupby('Year')['Total_Sales'].idxmax()
result = df_a.loc[max_sales_idx, ['Year', 'Total_Sales', 'Name','Genre']]
# %%
plt.figure(figsize=(12,8))
sns.lineplot(data=result, x= 'Year', y='Total_Sales', marker='o', hue='Name', legend='full')
plt.title('Each year best Game')
plt.xlabel('Year')
plt.ylabel('Total Sales')
plt.xticks(result['Year'])
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout()
plt.show()
# %%
df_a.info()
# %% 시각화는 Tableau로 하자

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %%
features = ['Platform', 'Genre', 'Publisher']
target = 'Total_Sales'
X = df_a[features]
y = df_a[target]

# 특성 데이터 인코딩
X_encoded = pd.get_dummies(X)

# 훈련 세트와 테스트 세트로 데이터 나누기
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 랜덤 포레스트 모델 생성 및 학습
rf = RandomForestRegressor(n_estimators=100, random_state=42)  # 트리의 개수는 100개로 설정했어요.
rf.fit(X_train, y_train)

# 테스트 세트로 예측
predictions = rf.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
# %%
