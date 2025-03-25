from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

def decisionTreeViz(df, label_col):
    # Mã hóa dữ liệu phân loại thành số
    label_encoders = {}
    
    for feature in df.columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le


    X = df.drop(columns=[label_col])
    y = df[label_col]
    # print(X)

    # Train ID3 Decision Tree (entropy = ID3 criterion)
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X, y)

    # Plot the tree
    # plt.figure(figsize=(9, 6))
    # plot_tree(model, filled=True, feature_names=X.columns, class_names=["Yes", "No"])
    # plt.show()

    return model, label_encoders






