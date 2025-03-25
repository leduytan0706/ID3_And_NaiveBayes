import math
import pandas as pd
import numpy as np
from typing import List, Union

class TreeNode:
    def __init__(self, attribute=None, value=None, label=None):
        self.attribute = attribute  # Thuộc tính kiểm tra
        self.value = value  # Giá trị của thuộc tính cha
        self.label = label  # Nhãn lớp (nếu là nút lá)
        self.children = {}  # Từ điển các nút con {giá trị thuộc tính: nút con}

class ID3DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.root = None
        self.features = None

    def entropy(self, series: pd.Series) -> float:
        # Tính entropy của tập dữ liệu
        class_counts = series.value_counts(normalize=True)
        entropy = -np.sum(class_counts * np.log2(class_counts))
        return entropy

    def information_gain(self, df: pd.DataFrame, feature: str, target: str) -> float:
        # Tính Information Gain: lượng thông tin được giữ lại
        # Tính entropy ban đầu
        base_entropy = self.entropy(df[target])
        
        # Chia tập dữ liệu theo các giá trị của thuộc tính
        weighted_entropy = 0
        for value in df[feature].unique():
            # Lọc các instance có giá trị thuộc tính này
            subset = df[df[feature] == value]
            # Tính entropy của tập con từng thuộc tính
            subset_entropy = self.entropy(subset[target])
            
            # Tính trọng số
            weight = len(subset) / len(df)
            weighted_entropy += weight * subset_entropy
        
        # Tính Information Gain
        return base_entropy - weighted_entropy

    def build_tree(self, df: pd.DataFrame, features: List[str], 
                   target: str, depth: int = 0) -> Union[TreeNode, str]:
        # Kiểm tra điều kiện dừng
        if len(df[target].unique()) == 1:
            return TreeNode(label=df[target].iloc[0])
        
        # Kiểm tra hết thuộc tính hoặc đạt độ sâu tối đa
        if (not features or 
            (self.max_depth is not None and depth >= self.max_depth)):
            return TreeNode(label=df[target].mode()[0])
        
        # Chọn thuộc tính phân chia tốt nhất (thuộc tính kiểm tra)
        best_feature = max(
            features, 
            key=lambda feat: self.information_gain(df, feat, target)
        )
        
        # Tạo nút gốc
        root = TreeNode(attribute=best_feature)
        
        # Loại bỏ thuộc tính kiểm tra
        remaining_features = [feat for feat in features if feat != best_feature]
        
        # Tạo các nhánh con
        for value in df[best_feature].unique():
            # Lọc ra tập con chứa thuộc tính 
            subset = df[df[best_feature] == value]
            
            # Nếu tập con rỗng, tạo nút lá với nhãn phổ biến nhất
            if len(subset) == 0:
                root.children[value] = TreeNode(
                    label=df[target].mode()[0]
                )
            else:
                # Đệ quy xây dựng nhánh con
                root.children[value] = self.build_tree(
                    subset, 
                    remaining_features, 
                    target, 
                    depth + 1
                )
        
        return root

    def fit(self, df: pd.DataFrame, target: str):
        # Huấn luyện cây quyết định
        # Lấy danh sách các thuộc tính (trừ thuộc tính mục tiêu)
        features = [col for col in df.columns if col != target]
        
        # Lưu lại danh sách features để sử dụng sau này
        self.features = features
        
        # Xây dựng cây
        self.root = self.build_tree(df, features, target)

    def predict(self, instance: pd.Series) -> Union[str, int]:
        # Dự đoán cho một mẫu mới
        node = self.root
        while node.attribute is not None:
            # Lấy giá trị của thuộc tính hiện tại
            attribute_value = instance.get(node.attribute)
            
            # Nếu giá trị không tồn tại trong cây, chọn nhánh phổ biến nhất
            if attribute_value not in node.children:
                child_nodes = list(node.children.values())
                node = child_nodes[0]  # Chọn nhánh đầu tiên
            else:
                # Di chuyển xuống nhánh con
                node = node.children[attribute_value]
        
        return node.label

    def predict_df(self, df: pd.DataFrame) -> pd.Series:
        """
        Dự đoán nhãn cho toàn bộ DataFrame
        """
        return df.apply(self.predict, axis=1)

    def print_tree(self, node: TreeNode = None, indent: str = ""):
        """
        In cây quyết định
        """
        if node is None:
            node = self.root
        
        # Nếu là nút lá
        if node.attribute is None:
            print(f"{indent}Leaf: {node.label}")
            return
        
        # In thông tin nút
        print(f"{indent}Attribute: {node.attribute}")
        
        # Duyệt các nhánh con
        for value, child in node.children.items():
            print(f"{indent}  ├── Value: {value}")
            self.print_tree(child, indent + "  │   ")

    def accuracy(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        """
        Tính độ chính xác của mô hình
        """
        predictions = self.predict_df(X_test)
        return (predictions == y_test).mean()

# Ví dụ sử dụng
def build_id3_tree(df, label):

    # Tách dữ liệu train và test
    from sklearn.model_selection import train_test_split
    
    X = df.drop(label, axis=1)
    y = df[label]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Tạo và huấn luyện cây quyết định
    tree = ID3DecisionTree(max_depth=3)
    tree.fit(pd.concat([X_train, y_train], axis=1), label)

    # In cây quyết định
    print("Cây quyết định:")
    tree.print_tree()

    # Kiểm tra dự đoán và độ chính xác
    # print("\nDự đoán:")
    predictions = tree.predict_df(X_test)
    # print("Predictions:", predictions.tolist())
    # print("Actual:", y_test.tolist())
    
    # Tính độ chính xác
    print(f"\nĐộ chính xác: {tree.accuracy(X_test, y_test)*100:.2f}%")

    return tree
    
