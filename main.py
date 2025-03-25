import pandas as pd
from ID3.dtree_build import decisionTreeViz
from NB.naive_bayes import NaiveBayes, predict_from_model
from sklearn.preprocessing import LabelEncoder
from basic_ID3.basic_ID3 import build_id3_tree


print("-------------------------------ID3-------------------------------")
# Bài 1
df = pd.DataFrame({
    "Day": ["D"+str(i+1) for i in range(14)],
    "Outlook": ["Sunny","Sunny","Overcast","Rain","Rain","Rain","Overcast","Sunny","Sunny","Rain","Sunny","Overcast","Overcast","Rain"],
    "Temperature": ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool","Mild","Mild","Mild","Hot","Mild"],
    "Humidity": ["High","High","High","High","Normal","Normal","Normal","High","Normal","Normal","Normal","High","Normal","High"],
    "Wind": ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Strong"],
    "Play_Tennis": ["No", "No", "Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
})


tree = build_id3_tree(df.drop(columns=["Day"]), "Play_Tennis")
new_sample = pd.Series({
    "Outlook": "Sunny",
    "Temperature": "Mild",
    "Humidity": "Normal",
    "Wind": "Weak"
})

print("\nDự đoán cho sample mới:")
print(new_sample)
print("Kết quả:", tree.predict(new_sample))



# Bài 2
print("-------------------------------Naive Bayes-------------------------------")

df2 = pd.DataFrame({
    "RecordId": [i+1 for i in range(14)],
    "Age": ["Young", "Young", "Medium", "Old", "Old", "Old", "Medium","Young","Young","Old","Young","Medium","Medium","Old"],
    "Income": ["High","High","High","Medium","Low","Low","Low","Medium","Low","Medium","Medium","Medium","High","Medium"],
    "Student": ["No","No","No","No","Yes","Yes","Yes","No","Yes","Yes","Yes","No","Yes","No"],
    "Credit_Rating": ["Fair","Excellent","Fair","Fair","Fair","Excellent","Excellent","Fair","Fair","Fair","Excellent","Excellent","Fair","Excellent"],
    "Buy_Computer": ["No","No","Yes","Yes","Yes","No","Yes","No","Yes","Yes","Yes","Yes","Yes","No"]
})

model2, label_encoders2 = NaiveBayes(df2.drop(columns=["RecordId"]), "Buy_Computer")
new_sample2 = pd.DataFrame({
    "Age": ["Young"],
    "Income": ["Medium"],
    "Student": ["No"],
    "Credit_Rating": ["Fair"]
})
print(new_sample)
pred2 = predict_from_model(new_sample2, "Buy_Computer", model2, label_encoders2)
print("Dự đoán khả năng mua máy tính:",pred2)


