import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest

def preprocess(file_path):
    data = pd.read_csv(file_path)

    residence_dummies = pd.get_dummies(data['Place_of_Residence'], prefix='Place', drop_first=False)
    data = pd.concat([data, residence_dummies], axis=1)
    
    data = data.drop(columns=['ID_Number', 'Education_Level', 'Is_Married', 'Place_of_Residence'])

    data['Years_of_Experience'] = np.log1p(data['Years_of_Experience'])
    
    experience_weight = data['Years_of_Experience'].copy()
    experience_weight[data['Years_of_Experience'] > np.log1p(10)] *= 2.5  # 10 yıl üzeri için %150 bonus
    
    residence_columns = [col for col in data.columns if col.startswith('Place_')]
    residence_score = data[residence_columns].mean(axis=1)
    
    data['economic_score'] = (
        (experience_weight * 0.35) +              
        ((1 - data['Is_House_Rent']) * 0.25) +    
        (data['Has_Car'] * 0.20) +                
        (-data['Number_of_Kids'] * 0.10) +        
        (residence_score * 0.10)                  
    )

    numerical_columns = ['Years_of_Experience', 'Number_of_Kids', 'economic_score']
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

    preprocessed_file_path = 'preprocessed_data.csv'
    data.to_csv(preprocessed_file_path, index=False)
    
    print(f"Preprocessed data saved at: {preprocessed_file_path}")
    return preprocessed_file_path

def train_model(preprocessed_file_path):

    data = pd.read_csv(preprocessed_file_path)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    isolation_forest = IsolationForest(contamination=0.3, random_state=42)  
    mask = isolation_forest.fit_predict(scaled_data) > 0  
    cleaned_data = scaled_data[mask]

    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(cleaned_data)

    n_clusters = 5
    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++", 
        n_init=10,        
        max_iter=400,     
        random_state=42
    )
    kmeans.fit(reduced_data)

    silhouette_avg = silhouette_score(reduced_data, kmeans.labels_)

    return {
        'centroids': kmeans.cluster_centers_,
        'labels': kmeans.labels_,
        'inertia': kmeans.inertia_,
        'silhouette_score': silhouette_avg
    }

data=preprocess("./credit_score_dataset.csv")

results = train_model('./preprocessed_data.csv')
print("Silhouette Score:", results['silhouette_score'])

data = pd.read_csv('./preprocessed_data.csv')
