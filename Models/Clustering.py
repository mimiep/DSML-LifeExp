#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    silhouette_samples,
    confusion_matrix
)
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from scipy.stats import mode

#For Print
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

#Read File
file_path =  r"Data_Cleaned.csv"
df = pd.read_csv(file_path, sep=",", quotechar='"', header=0)
print('Complete DataSet:')
print(df.head())

#Categories
categorical_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing"]

for col in categorical_columns:
    df[col] = df[col].astype("category")

print('Categorical:')
print(df.dtypes)
print()

#-----------------------------------
#Drop: Country, Region, Life_Expectancy,(Year), (Developed/Developing) ?
#Target: Region

# --------------------------------------------
# Region as Ground Truth
# --------------------------------------------
true_labels = df["Region"].cat.codes.values
region_names = df["Region"].cat.categories

# --------------------------------------------
# Drop unwanted columns
# --------------------------------------------
drop_columns = ["Country", "Region", "Economy_status_Developed", "Economy_status_Developing", "Year", "Hepatitis_B", "Diphtheria", "Measles", "Under_five_deaths", "Thinness_five_nine_years", "GDP_per_capita"]
df_features = df.drop(columns=drop_columns)

#Behalten	Droppen (redundant)
#Life_expectancy	–
#Adult_mortality	–
#Schooling	–
#GDP_per_capita	–
#BMI	–
#Incidents_HIV	–
#Polio - Hepatitis_B, Diphtheria, Measles
#Infant_deaths - Under_five_deaths
#Thinness_10_19	- Thinness_5_9
#Alcohol_consumption - optional – je nach Streuung
#Population_mln	optional – oft nicht sehr trennscharf

# --------------------------------------------
# Scale Features
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_features)

# --------------------------------------------
# Clusteranzahl k = number of unique regions
# --------------------------------------------
n_clusters = df["Region"].nunique()
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=50)
clusters = kmeans.fit_predict(X_scaled)

# --------------------------------------------
# Clustering Evaluation
# --------------------------------------------
ari = adjusted_rand_score(true_labels, clusters)
silhouette = silhouette_score(X_scaled, clusters)

print(f"Adjusted Rand Index (Region vs Cluster): {ari:.3f}")
print(f"Silhouette Score: {silhouette:.3f}")

# --------------------------------------------
# Mapping Cluster to Majority Region Label
# --------------------------------------------
def map_clusters_to_regions(true_labels, predicted_clusters):
    label_mapping = {}
    for cluster in np.unique(predicted_clusters):
        mask = predicted_clusters == cluster
        majority_region = mode(true_labels[mask], keepdims=True).mode[0]
        label_mapping[cluster] = majority_region
    return label_mapping

mapping = map_clusters_to_regions(true_labels, clusters)
mapped_predictions = np.array([mapping[c] for c in clusters])
accuracy = np.mean(mapped_predictions == true_labels)
print(f"Clustering Accuracy (best-matched): {accuracy:.2%}")

# --------------------------------------------
# t-SNE Projection for Cluster Visualization
# --------------------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df["Region"], style=clusters, palette="tab10", s=100)
plt.title("k-Means Clustering vs. Region (t-SNE Projection)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------
# Confusion Matrix (mapped clusters vs true labels)
# --------------------------------------------
cm = confusion_matrix(true_labels, mapped_predictions)

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=region_names,
            yticklabels=region_names)
plt.xlabel("Vorhergesagte Region (nach Mapping)")
plt.ylabel("Tatsächliche Region")
plt.title("Matching Accuracy Matrix (Cluster → Region)")
plt.tight_layout()
plt.show()

#good evaluation : africa, asia, european union, south america
#bad evaluation : central america and caribbean, north america, oceania, rest of europe
#moderat evaluation: middle east, south america

# --------------------------------------------
# Cluster Profile: Mean Feature Values by Cluster
# --------------------------------------------
df_features["Cluster"] = clusters
cluster_means = df_features.groupby("Cluster").mean()

# Zuordnung der Cluster zur jeweils häufigsten Region (für Label)
df_with_cluster = df.copy()
df_with_cluster["Cluster"] = clusters
region_labels = df_with_cluster.groupby("Cluster")["Region"].agg(lambda x: x.mode().iloc[0])
column_labels = [f"{cid} ({region_labels[cid]})" for cid in cluster_means.index]

# Heatmap mit Region-Namen im X-Achsenlabel
plt.figure(figsize=(14, 6))
sns.heatmap(cluster_means.T, cmap="vlag", annot=True, fmt=".1f",
            xticklabels=column_labels)
plt.title("Cluster Profile (Merkmalsmittelwerte mit Regionenzuordnung)")
plt.xlabel("Cluster (häufigste Region)")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --------------------------------------------
# Silhouette Score Histogramm
# --------------------------------------------
sample_silhouette = silhouette_samples(X_scaled, clusters)

plt.figure(figsize=(10, 6))
sns.histplot(sample_silhouette, bins=30, kde=True)
plt.axvline(silhouette, color="red", linestyle="--", label=f"Average = {silhouette:.2f}")
plt.title("Silhouette Scores Verteilung")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Anzahl Länder")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#average lays by ~0.24 = moderate speration of clusters

# --------------------------------------------
# Analyse gemeinsamer Cluster-Zugehörigkeit
# --------------------------------------------
from collections import defaultdict

region_similarity = defaultdict(int)
region_cluster_pairs = list(zip(df_with_cluster["Region"], df_with_cluster["Cluster"]))

for i in range(len(region_cluster_pairs)):
    for j in range(i + 1, len(region_cluster_pairs)):
        reg1, c1 = region_cluster_pairs[i]
        reg2, c2 = region_cluster_pairs[j]
        if c1 == c2:
            region_similarity[(reg1, reg2)] += 1

similar_list = sorted(region_similarity.items(), key=lambda x: -x[1])
print("\nRegionen, die häufig zusammen geclustert wurden:")
for pair, count in similar_list[:10]:
    print(f"{pair[0]} & {pair[1]} → gemeinsam in {count} Ländern")

#European Union & Rest of Europe → gemeinsam in 28121 Ländern
#Rest of Europe & European Union → gemeinsam in 27481 Ländern
#Central America and Caribbean & South America → gemeinsam in 14985 Ländern
#Africa & Asia → gemeinsam in 12537 Ländern
#------------------------------------------------------------------------------------------------------------------------
#following features droped : "Country", "Region", "Economy_status_Developed", "Economy_status_Developing", "Year", "Hepatitis_B", "Diphtheria", "Measles", "Under_five_deaths", "Thinness_five_nine_years"
#---------------------------------------------------------------------------------------------------------------------------

# --------------------------------------------
# HIERARCHISCHES CLUSTERING
# --------------------------------------------

print("\n--- Hierarchisches Clustering (Agglomerative) ---")

# Clustern
agglo = AgglomerativeClustering(n_clusters=n_clusters)  # gleiche Clusteranzahl wie k-Means
clusters_agglo = agglo.fit_predict(X_scaled)

# Bewertung
ari_agglo = adjusted_rand_score(true_labels, clusters_agglo)
silhouette_agglo = silhouette_score(X_scaled, clusters_agglo)

print(f"Adjusted Rand Index (Region vs Cluster): {ari_agglo:.3f}")
print(f"Silhouette Score: {silhouette_agglo:.3f}")

# Mapping
def map_clusters(true_labels, predicted_clusters):
    mapping = {}
    for cluster in np.unique(predicted_clusters):
        mask = predicted_clusters == cluster
        majority = mode(true_labels[mask], keepdims=True).mode[0]
        mapping[cluster] = majority
    return np.array([mapping[c] for c in predicted_clusters])

mapped_agglo = map_clusters(true_labels, clusters_agglo)
accuracy_agglo = np.mean(mapped_agglo == true_labels)
print(f"Clustering Accuracy (best-matched): {accuracy_agglo:.2%}")

#no big diference to normal clustering (slightly worse)
#ARI-> 0.17 worse
#Silhouette Score: 0.32 worse
#Accuracy: 0.03 worse

#-----------------------------------------------------------------
# Confusion Matrix: Hierarchical Clustering
#----------------------------------------------------------------

cm_agglo = confusion_matrix(true_labels, mapped_agglo)

plt.figure(figsize=(10, 6))
sns.heatmap(cm_agglo, annot=True, fmt="d", cmap="Blues",
            xticklabels=region_names, yticklabels=region_names)
plt.xlabel("Vorhergesagte Region (nach Mapping)")
plt.ylabel("Tatsächliche Region")
plt.title("Agglomerative Clustering: Matching Matrix")
plt.tight_layout()
plt.show()

#essentially the same as normal clustering
#only difference in true classification: central america and caribbean a lot better, south america a lot worse

#--------------------------------------------------------------------
# t-SNE for Hierachical Clustering
#--------------------------------------------------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne_agglo = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne_agglo[:, 0], y=X_tsne_agglo[:, 1], hue=df["Region"], style=clusters_agglo, palette="tab10", s=100)
plt.title("Agglomerative Clustering vs. Region (t-SNE Projection)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------
# Hyperparameter-Tuning: Optimal count of clusters (k)
# --------------------------------------------

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertias = []
silhouettes = []
k_range = range(2, 15)  # Teste k von 2 bis 14

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = kmeans.fit_predict(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

#-----------------------------------------------------------------
# Plot: Elbow-Method (Inertia)
#-----------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(k_range, inertias, marker="o")
plt.title("Elbow-Methode zur Auswahl von k")
plt.xlabel("Anzahl Cluster (k)")
plt.ylabel("Inertia (SSD)")
plt.grid(True)
plt.tight_layout()
plt.show()

#shows k= 5 would be best

#-----------------------------------------------------------------
# Plot: Silhouette Score
#-----------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(k_range, silhouettes, marker="s", color="green")
plt.title("Silhouette Score für verschiedene k")
plt.xlabel("Anzahl Cluster (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()
plt.show()

# Beste Wahl nach Silhouette
best_k = k_range[np.argmax(silhouettes)]
print(f"Beste Clusteranzahl laut Silhouette-Score: k = {best_k} (Score = {max(silhouettes):.3f})")

#silhoute score wise k= 3 would be best

#----------------------------------------------------------------------
# Comparison k = 3 vs k = 5
#----------------------------------------------------------------------
for test_k in [3, 5]:
    print(f"\n--- Modellbewertung für k = {test_k} ---")
    kmeans = KMeans(n_clusters=test_k, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)

    ari = adjusted_rand_score(true_labels, cluster_labels)
    silhouette = silhouette_score(X_scaled, cluster_labels)

    print(f"ARI (vs Region): {ari:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")

    # Mapping zur Region und Accuracy
    mapping = map_clusters_to_regions(true_labels, cluster_labels)
    mapped_preds = np.array([mapping[c] for c in cluster_labels])
    acc = np.mean(mapped_preds == true_labels)
    print(f"Matching Accuracy: {acc:.2%}")


# --------------------------------------------
# Final Analysis with k = 5
# --------------------------------------------

final_k = 5
kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=50)
clusters_final = kmeans_final.fit_predict(X_scaled)

# Evaluation
ari_final = adjusted_rand_score(true_labels, clusters_final)
silhouette_final = silhouette_score(X_scaled, clusters_final)
print(f"\n--- Finale Bewertung (k = {final_k}): ---")
print(f"ARI (Region vs Cluster): {ari_final:.3f}")
print(f"Silhouette Score: {silhouette_final:.3f}")

# Mapping zu Region
mapping_final = map_clusters_to_regions(true_labels, clusters_final)
mapped_final = np.array([mapping_final[c] for c in clusters_final])
accuracy_final = np.mean(mapped_final == true_labels)
print(f"Matching Accuracy: {accuracy_final:.2%}")

#---------------------------------------------------------------------
# Confusion Matrix k=5
#--------------------------------------------------------------------
cm_final = confusion_matrix(true_labels, mapped_final)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_final, annot=True, fmt="d", cmap="Blues",
            xticklabels=region_names, yticklabels=region_names)
plt.xlabel("Vorhergesagte Region (nach Mapping)")
plt.ylabel("Tatsächliche Region")
plt.title(f"Confusion Matrix für k = {final_k}")
plt.tight_layout()
plt.show()

#commonly misclassified regions: middle east, north america, oceania, rest of europe, south america

#----------------------------------------------------------------
# t-SNE Visualisierung (k=5)
#----------------------------------------------------------------
# Berechne t-SNE-Projektion für finale Cluster
tsne_final = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
X_tsne_final = tsne_final.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne_final[:, 0], y=X_tsne_final[:, 1],
                hue=clusters_final.astype(str), palette="Set2", s=100)
plt.title(f"k-Means Clustering Ergebnis (k = {final_k}) – t-SNE Projektion")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualisiert die Verteilung der 5 Cluster im Datenraum.
# Gute Trennung deutet auf sinnvolle Clusterung hin.
# Hier moderate Trennung, teils überlappende Strukturen sichtbar.

#------------------------------------------------------------------
# Heatmap: Clusterprofil
#------------------------------------------------------------------
df_features["Cluster_k5"] = clusters_final
cluster_means_k5 = df_features.groupby("Cluster_k5").mean()
plt.figure(figsize=(14, 6))
sns.heatmap(cluster_means_k5.T, cmap="vlag", annot=True, fmt=".1f")
plt.title("Clusterprofil für k=5")
plt.xlabel("Cluster")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

#Cluster 0 : high Polio/life_expectancy/schooling/alcohol_consumption, low infant_deaths/thinnes_ten_nineteen_years
#Cluster 1 : high adult_mortality(very high)/Polio/ infant_deaths/ incidents_hiv, low life_expectancy
#Cluster 2 : high BMI/Polio/Life_expectancy, low thinnes_ten_nineteen_years/ alcohol_consumption
#Cluster 3 : high adult_mortality/ infant_deaths/ population_min, low alcohol_consumption/ schooling/ life_expectancy
#Cluster 4 : high population_min(very high)/ thinnes_ten_nineteen_years, low incidents_hiv/ BMI

#-------------------------------------------------------------------
# Heatmap: Tatsächliche Regionen vs. zugewiesene Cluster (k=5)
#------------------------------------------------------------------
region_cluster_table = pd.crosstab(df["Region"], clusters_final)

plt.figure(figsize=(12, 6))
sns.heatmap(region_cluster_table, annot=True, fmt="d", cmap="YlGnBu")
plt.title(f"Regionen pro Cluster – Häufigkeitstabelle (k = {final_k})")
plt.xlabel("Cluster")
plt.ylabel("Tatsächliche Region")
plt.tight_layout()
plt.show()

#Cluster 0: high european union/ rest of europ/ central america and caribbean
#Cluster 1: high africa
#cluster 2: high asia/ central america and caribbean/ middle east/ south america
#cluster 3: high africa/ asia
#cluster 4: asia




#SIMON