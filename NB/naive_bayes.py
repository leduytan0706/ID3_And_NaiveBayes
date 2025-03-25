import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

def NaiveBayes(df, label_col):
    # Mã hóa dữ liệu phân loại thành số
    label_encoders = {}
    for feature in df.columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
        label_encoders[feature] = le  # Lưu bộ mã hóa để giải mã sau này


    X = df.drop(columns=[label_col])
    y = df[label_col]
    # Chia tập dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện mô hình Naive Bayes
    model = CategoricalNB()
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác: {accuracy * 100:.2f}%")

    return model, label_encoders

def predict_from_model(new_sample, label_col, model, label_encoders):
    for feature in new_sample.columns:
        new_sample[feature] = label_encoders[feature].transform(new_sample[feature])

    prediction = model.predict(new_sample)
    result = label_encoders[label_col].inverse_transform(prediction)[0]
    return result






