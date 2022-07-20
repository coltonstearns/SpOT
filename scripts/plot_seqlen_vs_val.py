import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data_file = "/home/colton/Downloads/wandb_export_2022-05-24T13_09_59.318-07_00.csv"
df = pd.read_csv(data_file)[:-1]
print(df)
print(df.columns)

sns.lineplot(data=df, x='training_dataloading.sequence-properties.sequence-length', y="val: metric yaw-l1 mean")
# sns.lineplot(data=df, x='training_dataloading.sequence-properties.sequence-length', y="val: metric velocity-l1 mean")
plt.show()