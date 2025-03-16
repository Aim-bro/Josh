# %%
from naver_map_data import scrape_places
import pandas as pd

keyword = "강남역 맛집"
data = scrape_places(keyword)
df = pd.DataFrame(data)
#%%
print(df)
# %%
