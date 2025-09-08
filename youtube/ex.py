# %%
import pandas as pd
df = pd.read_csv('data/meta/hwi_videos_labeled.csv')
df_text = df[['title', 'label_rule']]

# %%
print(df_text)
# %%
