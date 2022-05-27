import matplotlib.pyplot as plt
import pandas as pd

base = pd.read_csv("C:/Users/FerGo/OneDrive/ACIT/01.2022/Data Mining/Project/checkpoints/results.csv",
                   index_col=0)


plt.figure()
plt.plot(base["training loss"])
plt.plot(base["validation loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()

plt.figure()
plt.plot(base["training accuracy"])
plt.plot(base["validation accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "test"], loc="upper left")
plt.show()
