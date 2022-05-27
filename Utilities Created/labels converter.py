import pandas as pd


base = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/Data_Entry_2017_v2020.csv",
                   index_col=(0))

labels = pd.DataFrame(base["Finding Labels"])
expanded_labels = labels["Finding Labels"].str.split("|", expand = True)
all_labels = pd.DataFrame(expanded_labels[[0,1,2,3,4,5,6,7,8]].values.ravel("k"), columns={"Labels"})
unique_labels = all_labels["Labels"].unique()
unique_labels = unique_labels[:-1]

order = [1,0,2,3,4,5,6,7,8,9,10,11,12,13,14]
unique_labels = [unique_labels[x] for x in order]


new_labels = pd.DataFrame(0, index = base.index, columns = unique_labels)

for label in unique_labels:
    temp = expanded_labels[expanded_labels == label].dropna(how = "all")
    new_labels[label] = new_labels.index.isin(temp.index).astype(int)

new_labels.to_csv("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/new_labels.csv")
