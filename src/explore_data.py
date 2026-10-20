
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/creditcard.csv")

class_counts = df["Class"].value_counts().sort_index()

print("Normal transactions:", class_counts[0])
print("Fraudulent transactions:", class_counts[1])
print("Fraud percentage:", round((class_counts[1] / len(df)) * 100, 4), "%")

plt.figure(figsize=(8, 5))
sns.barplot(x=class_counts.index, y=class_counts.values)

plt.title("Class Distribution: Normal vs Fraud")
plt.xlabel("Class (0 = Normal, 1 = Fraud)")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.tight_layout()
plt.savefig("reports/class_distribution.png", dpi=300)
plt.show()
