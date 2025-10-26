import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, silhouette_score
import numpy as np

df = pd.read_csv('tip.csv')
df.dropna(inplace=True)

le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_day = LabelEncoder()
le_time = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'])
df['smoker'] = le_smoker.fit_transform(df['smoker'])
df['day'] = le_day.fit_transform(df['day'])
df['time'] = le_time.fit_transform(df['time'])

features = ['total_bill','sex','smoker','day','time','size']
X = df[features]
y_tip = df['tip']
y_size = df['size']

# Supervised Regression
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_tip, test_size=0.2, random_state=42)
regressor = LinearRegression()
regressor.fit(X_train_r, y_train_r)
y_pred_r = regressor.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)

# Supervised Classification
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_size, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
classifier.fit(X_train_c, y_train_c)
y_pred_c = classifier.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred_c)

# Unsupervised Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(X)
silhouette = silhouette_score(X, df['cluster'])

# Unsupervised PCA
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)
df_pca = pd.DataFrame(data=pca_components, columns=['PC1','PC2'])

# Determine best algorithm
best_algorithm = max([('Regression', 1/(mse+1e-5)), ('Classification', accuracy), ('Clustering', silhouette)],
                     key=lambda x: x[1])[0]

plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.scatter(y_test_r, y_pred_r, color='green')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], color='red')
plt.title(f'Regression: Tip Prediction (MSE: {mse:.2f})')
plt.xlabel('Actual Tip')
plt.ylabel('Predicted Tip')

plt.subplot(2,2,2)
plt.bar(['Random Forest'], [accuracy*100], color='blue')
plt.ylim(0,100)
plt.title(f'Classification: Size Prediction (Accuracy: {accuracy*100:.2f}%)')
plt.ylabel('Accuracy (%)')

plt.subplot(2,2,3)
plt.scatter(df['total_bill'], df['tip'], c=df['cluster'], cmap='viridis')
plt.title(f'Clustering: Tips by Total Bill (Silhouette: {silhouette:.2f})')
plt.xlabel('Total Bill')
plt.ylabel('Tip')

plt.subplot(2,2,4)
plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df['cluster'], cmap='viridis')
plt.title('PCA Visualization')
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.tight_layout()
plt.show()

print("Performance Metrics:")
print(f"- Regression (Tip Prediction) MSE: {mse:.4f} (Lower is better)")
print(f"- Classification (Size Prediction) Accuracy: {accuracy*100:.2f}%")
print(f"- Clustering (KMeans) Silhouette Score: {silhouette:.4f}")
print(f"\nBest Performing Algorithm: {best_algorithm}")
