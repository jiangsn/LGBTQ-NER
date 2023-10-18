# %% load libraries
import numpy as np
import pandas as pd
print(pd.__version__)


# %% 
df = pd.read_xml('./MeSH/desc2022.xml')
print(len(df))
df.tail(20)





# %%
