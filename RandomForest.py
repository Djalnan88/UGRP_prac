import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score

file_path = 'dataset/Wednesday-workingHours.pcap_ISCX.csv'
data = pd.read_csv(file_path)

df = data.copy()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.columns = df.columns.str.strip().str.replace('[ /]', '_', regex=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

X = df.drop('Label', axis=1)
y = df['Label']
y_binary = np.where(y == 'BENIGN', 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.15, random_state=42, stratify=y_binary
)

print(f"Train 데이터 크기: {X_train.shape}")
print(f"Test 데이터 크기: {X_test.shape}")
print("-" * 50)

print("## 1. 전체 Feature를 사용한 모델 학습 및 평가 ##")
model_full = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model_full.fit(X_train, y_train)

y_pred_full = model_full.predict(X_test)
accuracy_full = accuracy_score(y_test, y_pred_full)
recall_full = recall_score(y_test, y_pred_full)
f1_full = f1_score(y_test, y_pred_full)

print(f"모델 정확도: {accuracy_full:.4f}")
print(f"재현율 (Recall): {recall_full:.4f}")
print(f"F1-Score: {f1_full:.4f}")
print("성능 리포트:")
print(classification_report(y_test, y_pred_full, target_names=['BENIGN (0)', 'ATTACK (1)']))
print("-" * 50)

print("## 2. Gini 중요도 기반 상위 10개 특성 추출 ##")
importances = model_full.feature_importances_
feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})
feature_importances = feature_importances.sort_values('importance', ascending=False)
top_10_feature_names = feature_importances.head(10)['feature'].tolist()

print("상위 10개 특성:")
print(feature_importances.head(10))
print("-" * 50)

print("## 3. 상위 10개 특성만 사용한 모델 학습 및 평가 ##")
X_train_top10 = X_train[top_10_feature_names]
X_test_top10 = X_test[top_10_feature_names]

model_top10 = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)
model_top10.fit(X_train_top10, y_train)

y_pred_top10 = model_top10.predict(X_test_top10)
accuracy_top10 = accuracy_score(y_test, y_pred_top10)
recall_top10 = recall_score(y_test, y_pred_top10)
f1_top10 = f1_score(y_test, y_pred_top10)

print(f"모델 정확도: {accuracy_top10:.4f}")
print(f"재현율 (Recall): {recall_top10:.4f}")
print(f"F1-Score: {f1_top10:.4f}")
print("성능 리포트:")
print(classification_report(y_test, y_pred_top10, target_names=['BENIGN (0)', 'ATTACK (1)']))
print("-" * 50)

print("## 4. 최종 성능 비교 ##")
print("[전체 특성 사용 모델]")
print(f" - 정확도 (Accuracy): {accuracy_full:.4f}")
print(f" - 재현율 (Recall): {recall_full:.4f}")
print(f" - F1-Score: {f1_full:.4f}")

print("[상위 10개 특성 사용 모델]")
print(f" - 정확도 (Accuracy): {accuracy_top10:.4f}")
print(f" - 재현율 (Recall): {recall_top10:.4f}")
print(f" - F1-Score: {f1_top10:.4f}")
