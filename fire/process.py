from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.table import table
from pandas.plotting import table
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os
import re

# Load dataset
data = pd.read_csv('./FT-121.csv')

# Cleaning up the dataset by dropping the Timestamp column and converting YES/NO to binary (1/0)
data_clean = data.drop(columns=['Timestamp'])
data_clean = data_clean.replace({'YES': 1, 'NO': 0})
data_clean.columns = data_clean.columns.str.extract(r'\[(.*?)\]')[0]

# Calculate the count of "Yes" and "No" responses for each question
yes_counts = data_clean.sum()
no_counts = data_clean.shape[0] - yes_counts

# Calculate the probabilities (percentages) of "Yes" and "No" responses
yes_probabilities = (yes_counts / data_clean.shape[0]) * 100
no_probabilities = (no_counts / data_clean.shape[0]) * 100

# Calculate the frequency of "Yes" (which is represented by 1) for each question
yes_frequencies = yes_counts.sort_values(ascending=False)

# Reorder the columns in the dataset according to the frequency of "Yes" answers
data_clean = data_clean[yes_frequencies.index]

# Create a DataFrame to hold the results
response_ratios = pd.DataFrame({
    'Yes Count': yes_counts,
    'No Count': no_counts,
    'Yes Probability (%)': yes_probabilities,
    'No Probability (%)': no_probabilities
})

# Sort the DataFrame by 'Yes Count' in descending order
response_ratios = response_ratios.sort_values(by='Yes Count', ascending=True)

# Create a horizontal bar chart with split bars for Yes and No counts, with questions on the y-axis
fig, ax = plt.subplots(figsize=(14, 16))  # Adjust figure size

# Plot the Yes counts in green
ax.barh(response_ratios.index, response_ratios['Yes Count'], color='green', label='Yes Count')

# Plot the No counts in red (stacked on top of yes counts
ax.barh(response_ratios.index, response_ratios['No Count'], left=response_ratios['Yes Count'], color='red', label='No Count')

# Add labels and title
ax.set_xlabel('Response Counts')
ax.set_title('Yes/No Response Counts per Question')
ax.legend()

# Adjust the layout to fit everything
plt.tight_layout()

# Save the chart as an image
chart_image_path = './yes_no_ratio_chart.png'
plt.savefig(chart_image_path)

# Create a new figure for the table
fig, ax = plt.subplots(figsize=(8, 6))  # Adjust figure size for table readability
ax.axis('off')  # Hide the axis

# sort yes to top
response_ratios = response_ratios.sort_values(by='Yes Count', ascending=False)

# Format the 'Yes Count' and 'No Count' columns as integers
response_ratios['Yes Count'] = response_ratios['Yes Count'].astype(int)
response_ratios['No Count'] = response_ratios['No Count'].astype(int)

# Format 'Yes Probability (%)' and 'No Probability (%)' to two decimal places
response_ratios['Yes Probability (%)'] = response_ratios['Yes Probability (%)'].round(2)
response_ratios['No Probability (%)'] = response_ratios['No Probability (%)'].round(2)

# Create the table from the response_ratios DataFrame
tbl = table(ax, response_ratios, loc='center', cellLoc='center', colWidths=[0.2] * len(response_ratios.columns))

# Customize table appearance
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)  # Scale table for better readability

# Save the table as an image
table_image_path = './yes_no_response_table.png'
plt.savefig(table_image_path, bbox_inches='tight')

# Function to calculate Jaccard index between two binary vectors
def jaccard_index(x, y):
    intersection = np.sum((x == 1) & (y == 1))
    union = np.sum((x == 1) | (y == 1))
    return intersection / union if union != 0 else 0

# Get the list of questions (column names)
questions = data_clean.columns

# Create an empty matrix to store Jaccard indices
jaccard_matrix = np.zeros((len(questions), len(questions)))

# Compute Jaccard index for every pair of questions
for i in range(len(questions)):
    for j in range(len(questions)):
        if i != j:
            jaccard_matrix[i, j] = jaccard_index(data_clean.iloc[:, i], data_clean.iloc[:, j])
        else:
            jaccard_matrix[i, j] = 1  # Jaccard index with itself is 1

# Create a heatmap of the Jaccard index matrix
plt.figure(figsize=(20, 16))
sns.heatmap(jaccard_matrix, xticklabels=questions, yticklabels=questions, cmap='coolwarm', annot=False)

plt.title("Jaccard Index Heatmap of Binary Responses")
plt.xticks(rotation=90)
plt.xlabel("Question B")
plt.ylabel("Question A")
plt.tight_layout()

# Save the heatmap as an image
heatmap_image_path = './jaccard_heatmap.png'
plt.savefig(heatmap_image_path)

# Function to calculate conditional probability P(B | A)
def conditional_prob(A, B):
    A_and_B = np.sum((A == 1) & (B == 1))  # Count of Yes to both A and B
    A_count = np.sum(A == 1)  # Count of Yes to A
    return A_and_B / A_count if A_count != 0 else 0

# Get the list of questions (column names)
questions = data_clean.columns

# Create an empty matrix to store conditional probabilities
conditional_prob_matrix = np.zeros((len(questions), len(questions)))

# Compute conditional probabilities P(B | A) for every pair of questions
for i in range(len(questions)):
    for j in range(len(questions)):
        if i != j:
            conditional_prob_matrix[i, j] = conditional_prob(data_clean.iloc[:, i], data_clean.iloc[:, j])
        else:
            conditional_prob_matrix[i, j] = 1  # P(A | A) is always 1

# Create a heatmap of the conditional probability matrix
plt.figure(figsize=(20, 16))
sns.heatmap(conditional_prob_matrix, xticklabels=questions, yticklabels=questions, cmap='coolwarm', annot=False)

plt.title("Conditional Probability Heatmap of Binary Responses (P(B | A))")
plt.xlabel("Question B")
plt.ylabel("Question A")
plt.xticks(rotation=90)
plt.tight_layout()

# Save the heatmap as an image
conditional_prob_heatmap_path = './conditional_prob_heatmap.png'
plt.savefig(conditional_prob_heatmap_path)

# Function to calculate joint probability P(A and B)
def joint_prob(A, B):
    A_and_B = np.sum((A == 1) & (B == 1))  # Count of Yes to both A and B
    total = len(A)  # Total number of samples
    return A_and_B / total if total != 0 else 0

# Create an empty matrix to store joint probabilities
joint_prob_matrix = np.zeros((len(questions), len(questions)))

# Compute joint probabilities P(A and B) for every pair of questions
for i in range(len(questions)):
    for j in range(len(questions)):
        joint_prob_matrix[i, j] = joint_prob(data_clean.iloc[:, i], data_clean.iloc[:, j])

# Create a heatmap of the joint probability matrix
plt.figure(figsize=(20, 16))
sns.heatmap(joint_prob_matrix, xticklabels=questions, yticklabels=questions, cmap='coolwarm', annot=False)

plt.title("Joint Probability Heatmap of Binary Responses (P(A and B))")
plt.xlabel("Question B")
plt.ylabel("Question A")
plt.xticks(rotation=90)
plt.tight_layout()

# Save the heatmap as an image
joint_prob_heatmap_path = './joint_prob_heatmap.png'
plt.savefig(joint_prob_heatmap_path)

# Create a DataFrame to store results of GMM or uh BMM i guess?
bmmResults = []

# Try different values of k (latent classes) from 2 to 20
for k in range(2, 21):
    # Fit a Gaussian Mixture Model as a proxy for Bernoulli Mixture
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(data_clean)
    
    # Record the AIC, BIC, and log likelihood
    aic = gmm.aic(data_clean)
    bic = gmm.bic(data_clean)
    log_likelihood = gmm.lower_bound_  # Log likelihood approximation
    
    # Append the metrics to results
    bmmResults.append({
        'Latent Classes (k)': k,
        'AIC': aic,
        'BIC': bic,
        'Log Likelihood': log_likelihood
    })

# Convert the results to a DataFrame
bmmResults_df = pd.DataFrame(bmmResults)

# Plot the AIC and BIC values for each k
plt.figure(figsize=(10, 6))
plt.plot(bmmResults_df['Latent Classes (k)'], bmmResults_df['AIC'], label='AIC', marker='o')
plt.plot(bmmResults_df['Latent Classes (k)'], bmmResults_df['BIC'], label='BIC', marker='o')
plt.xlabel('Number of Latent Classes (k)')
plt.ylabel('Metric Value')
plt.title('AIC and BIC for Different Numbers of Latent Classes (GMM Approximation)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image
plt.savefig('./gmm_aic_bic_plot.png')

# Compute pairwise Jaccard distances
jaccard_distances = pdist(data_clean, metric='jaccard')

# Convert the distances to a square matrix form
jaccard_matrix = squareform(jaccard_distances)

# Create a DataFrame to store results of kmeans
km_results = []

# Try different values of k (latent classes) from 2 to 20
for k in range(2, 21):
    # Fit KMeans on Jaccard distance matrix
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(jaccard_matrix)
    
    # Record inertia (sum of squared distances to closest cluster center)
    inertia = kmeans.inertia_
    
    # Append the results
    km_results.append({
        'Latent Classes (k)': k,
        'Inertia': inertia
    })

# Convert the results to a DataFrame
km_results_df = pd.DataFrame(km_results)

# Plot the Inertia values for each k
plt.figure(figsize=(10, 6))
plt.plot(km_results_df['Latent Classes (k)'], km_results_df['Inertia'], label='Inertia', marker='o')
plt.xlabel('Number of Latent Classes (k)')
plt.ylabel('Inertia')
plt.title('Inertia for Different Numbers of Latent Classes (K-Means with Jaccard Distance)')
plt.grid(True)
plt.tight_layout()

# Save the plot as an image
plt.savefig('./kmeans_inertia_plot.png')

# For GMM (Gaussian Mixture Model / BMM approximation)
def get_gmm_cluster_assignments(k, data):
    gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42)
    gmm.fit(data)
    return gmm.predict(data)

# For K-Means (using Jaccard Distance)
def get_kmeans_cluster_assignments(k, jaccard_matrix):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(jaccard_matrix)
    return kmeans.labels_

# Example: Get cluster assignments for k = 5, 6, 7, 8
cluster_results = {}

for k in [5, 6, 7, 8]:
    gmm_clusters = get_gmm_cluster_assignments(k, data_clean)
    kmeans_clusters = get_kmeans_cluster_assignments(k, jaccard_matrix)
    
    cluster_results[k] = {
        'GMM': gmm_clusters,
        'KMeans': kmeans_clusters
    }

# Convert to DataFrame for easy comparison
cluster_comparison_df = pd.DataFrame({
    'GMM_5': cluster_results[5]['GMM'],
    'KMeans_5': cluster_results[5]['KMeans'],
    'GMM_6': cluster_results[6]['GMM'],
    'KMeans_6': cluster_results[6]['KMeans'],
    'GMM_7': cluster_results[7]['GMM'],
    'KMeans_7': cluster_results[7]['KMeans'],
    'GMM_8': cluster_results[8]['GMM'],
    'KMeans_8': cluster_results[8]['KMeans']
})

# Dictionary to store heatmap data
heatmap_data = {}

# Loop through the chosen values of k and both clustering methods (GMM and KMeans)
for k in [5, 6, 7, 8]:
    # GMM Clusters
    gmm_clusters = cluster_results[k]['GMM']
    gmm_cluster_probs = []

    # For each cluster, calculate the probability of "Yes" for each question
    for cluster in np.unique(gmm_clusters):
        cluster_data = data_clean[gmm_clusters == cluster]
        yes_prob = cluster_data.mean()
        gmm_cluster_probs.append(yes_prob)

    # Create DataFrame for GMM clusters and sort columns by the sum of "Yes" probabilities
    gmm_probs_df = pd.DataFrame(gmm_cluster_probs).T
    gmm_probs_df.columns = [f'GMM_k{k}_Cluster{cluster}' for cluster in range(1, k + 1)]
    gmm_probs_df = gmm_probs_df.loc[:, gmm_probs_df.sum().sort_values(ascending=False).index]

    # Add NaN column to separate models
    gmm_probs_df['Spacer_GMM'] = np.nan

    # KMeans Clusters
    kmeans_clusters = cluster_results[k]['KMeans']
    kmeans_cluster_probs = []

    # For each cluster, calculate the probability of "Yes" for each question
    for cluster in np.unique(kmeans_clusters):
        cluster_data = data_clean[kmeans_clusters == cluster]
        yes_prob = cluster_data.mean()
        kmeans_cluster_probs.append(yes_prob)

    # Create DataFrame for KMeans clusters and sort columns by the sum of "Yes" probabilities
    kmeans_probs_df = pd.DataFrame(kmeans_cluster_probs).T
    kmeans_probs_df.columns = [f'KMeans_k{k}_Cluster{cluster}' for cluster in range(1, k + 1)]
    kmeans_probs_df = kmeans_probs_df.loc[:, kmeans_probs_df.sum().sort_values(ascending=False).index]

    # Add NaN column to separate groups in the heatmap
    kmeans_probs_df['Spacer_KMeans'] = np.nan

    # Combine GMM and KMeans data for this value of k
    combined_df = pd.concat([gmm_probs_df, kmeans_probs_df], axis=1)
    heatmap_data[k] = combined_df

# Combine all k-values into a single DataFrame for the heatmap
full_heatmap_df = pd.concat(heatmap_data.values(), axis=1)

# Calculate baseline probabilities (overall probability of 'Yes' for each question)
baseline_probs = data_clean.mean()

# Convert to DataFrame and add a spacer column
baseline_probs_df = pd.DataFrame(baseline_probs, columns=['Baseline'])
baseline_probs_df['Spacer_Baseline'] = np.nan

# Now, concatenate baseline_probs_df with full_heatmap_df, ensuring baseline comes first
full_heatmap_df_with_baseline = pd.concat([baseline_probs_df, full_heatmap_df], axis=1)

# Plot the heatmap with baseline included
plt.figure(figsize=(20, 14))
sns.heatmap(full_heatmap_df_with_baseline, cmap='coolwarm', annot=False, cbar=True)

# Configure heatmap labels
plt.title('Probability of "Yes" Responses by Baseline and Clusters for GMM and KMeans Models, Clustlets Sorted by Yes Probabilities')
plt.xlabel('Baseline and Clusters (Grouped by Model and k)')
plt.ylabel('Questions')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the updated heatmap
heatmap_with_baseline_image_path = './cluster_heatmap.png'
plt.savefig(heatmap_with_baseline_image_path)

# Define the list of high-risk behaviors
high_risk_behaviors = [
    'Burned items on a BBQ that weren\'t food.',
    'Started a fire accidently.',
    'Accidently started a fire in the house.',
    'Thrown or placed dangerous items into a fire.',
    'Had a fire get out of control.',
    'Tried to hide a fire that I started.',
    'Attempted to conceal a fire.',
    'Been in trouble because of fire.',
    'Started an extremely large fire.',
    'Lied about a fire I started.',
    'Used fire as a cry for help.',
    'Used fire to destroy or conceal evidence of a crime.',
    'Been in legal or criminal trouble because of a fire.',
    'Set a fire to get revenge.',
    'Hurt someone else with fire.',
    'Used fire to manipulate or control someone.',
    'Used fire as a weapon.',
    'Used fire to get attention.'
]

# Dictionary to store heatmap data sorted by high-risk behaviors
high_risk_sorted_heatmap_data = {}

# Loop through the chosen values of k and both clustering methods (GMM and KMeans)
for k in [5, 6, 7, 8]:
    # GMM Clusters
    gmm_clusters = cluster_results[k]['GMM']
    gmm_cluster_probs = []

    # For each cluster, calculate the probability of "Yes" for each question
    for cluster in np.unique(gmm_clusters):
        cluster_data = data_clean[gmm_clusters == cluster]
        yes_prob = cluster_data.mean()
        gmm_cluster_probs.append(yes_prob)

    # Create DataFrame for GMM clusters, sorting columns by the sum of "Yes" probabilities in high-risk behaviors
    gmm_probs_df = pd.DataFrame(gmm_cluster_probs).T
    gmm_probs_df.columns = [f'GMM_k{k}_Cluster{cluster}' for cluster in range(1, k + 1)]
    high_risk_sort_order = gmm_probs_df.loc[high_risk_behaviors].sum().sort_values(ascending=False).index
    gmm_probs_df = gmm_probs_df[high_risk_sort_order]

    # Add NaN column to separate models
    gmm_probs_df['Spacer_GMM_HighRiskSorted'] = np.nan

    # KMeans Clusters
    kmeans_clusters = cluster_results[k]['KMeans']
    kmeans_cluster_probs = []

    # For each cluster, calculate the probability of "Yes" for each question
    for cluster in np.unique(kmeans_clusters):
        cluster_data = data_clean[kmeans_clusters == cluster]
        yes_prob = cluster_data.mean()
        kmeans_cluster_probs.append(yes_prob)

    # Create DataFrame for KMeans clusters, sorting columns by the sum of "Yes" probabilities in high-risk behaviors
    kmeans_probs_df = pd.DataFrame(kmeans_cluster_probs).T
    kmeans_probs_df.columns = [f'KMeans_k{k}_Cluster{cluster}' for cluster in range(1, k + 1)]
    high_risk_sort_order_kmeans = kmeans_probs_df.loc[high_risk_behaviors].sum().sort_values(ascending=False).index
    kmeans_probs_df = kmeans_probs_df[high_risk_sort_order_kmeans]

    # Add NaN column to separate groups in the heatmap
    kmeans_probs_df['Spacer_KMeans_HighRiskSorted'] = np.nan

    # Combine GMM and KMeans data for this value of k
    combined_high_risk_sorted_df = pd.concat([gmm_probs_df, kmeans_probs_df], axis=1)
    high_risk_sorted_heatmap_data[k] = combined_high_risk_sorted_df

# Combine all k-values into a single DataFrame for the high-risk-sorted heatmap
full_high_risk_sorted_heatmap_df = pd.concat(high_risk_sorted_heatmap_data.values(), axis=1)

# Concatenate baseline probabilities and add a spacer column
baseline_probs_df_with_spacer = baseline_probs.to_frame(name='Baseline')
baseline_probs_df_with_spacer['Spacer_Baseline_HighRiskSorted'] = np.nan

# Now, concatenate baseline with the high-risk sorted heatmap, ensuring baseline comes first
full_high_risk_sorted_heatmap_df_with_baseline = pd.concat([baseline_probs_df_with_spacer, full_high_risk_sorted_heatmap_df], axis=1)

# Reorder rows: high-risk behaviors first, followed by the remaining behaviors
# Get the list of behaviors not in high-risk
non_high_risk_behaviors = [b for b in full_high_risk_sorted_heatmap_df_with_baseline.index if b not in high_risk_behaviors]

# Combine high-risk behaviors and the remaining behaviors in the desired order
sorted_row_order = high_risk_behaviors + non_high_risk_behaviors

# Reorder the DataFrame based on this new row order
full_high_risk_sorted_heatmap_df_with_baseline = full_high_risk_sorted_heatmap_df_with_baseline.loc[sorted_row_order]

# Plot the reordered heatmap with baseline included
plt.figure(figsize=(20, 14))
sns.heatmap(full_high_risk_sorted_heatmap_df_with_baseline, cmap='coolwarm', annot=False, cbar=True)

# Configure heatmap labels
plt.title('Probability of "Yes" Responses by Baseline and Clusters, Rows Sorted with High-Risk Behaviors First')
plt.xlabel('Baseline and Clusters (Grouped by Model and k)')
plt.ylabel('Questions')
plt.xticks(rotation=90)
plt.tight_layout()

# Save the reordered high-risk sorted heatmap
high_risk_sorted_heatmap_image_path = './cluster_heatmap_high_risk.png'
plt.savefig(high_risk_sorted_heatmap_image_path)

# note to choose k means k=6 since it highlights the extreme low fire interaction but also cry for help

# Create two groups based on high-risk behaviors
# Group with all 0 for high-risk behaviors
no_high_risk_group = data_clean[(data_clean[high_risk_behaviors].sum(axis=1) == 0)]

# Group with at least one 1 in high-risk behaviors
has_high_risk_group = data_clean[(data_clean[high_risk_behaviors].sum(axis=1) > 0)]

# Calculate the baseline probabilities across all data
baseline_probs = data_clean.mean()

# Calculate mean probabilities for both groups
no_high_risk_probs = no_high_risk_group.mean()
has_high_risk_probs = has_high_risk_group.mean()

# Create a DataFrame to hold these group probabilities
group_comparison_df = pd.DataFrame({
    'Baseline (%)': baseline_probs,
    'No High Risk (%)': no_high_risk_probs,
    'Has High Risk (%)': has_high_risk_probs
})

# Combine all columns for the heatmap with the baseline column first
heatmap_with_groups_df = group_comparison_df[['Baseline (%)', 'No High Risk (%)', 'Has High Risk (%)']]

# Plot the heatmap for group comparison with baseline included
plt.figure(figsize=(14, 14))
sns.heatmap(heatmap_with_groups_df, cmap='coolwarm', annot=False, cbar=True)

# Configure heatmap labels
plt.title('Probability of "Yes" Responses: Baseline vs. No High Risk Group vs. Has High Risk Group')
plt.xlabel('Groups')
plt.ylabel('Questions')
plt.xticks(rotation=45)
plt.tight_layout()

# Save the heatmap
group_comparison_heatmap_path = './group_comparison_heatmap.png'
plt.savefig(group_comparison_heatmap_path)

# create a table to show the heatmap data for the group comparison

# round all percents to 2 decimal points. and multiply them by 100 to make them percents
heatmap_with_groups_df = (heatmap_with_groups_df * 100).round(2)

# Create a table figure
fig, ax = plt.subplots(figsize=(8, 10))  # Adjust height based on the number of behaviors
ax.axis('off')  # Hide axes

# Add the table to the figure
tbl = table(ax, heatmap_with_groups_df, loc='center', cellLoc='center', colWidths=[0.2] * heatmap_with_groups_df.shape[1])

# Customize table appearance
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 1.2)  # Scale table for readability

# Save the table as an image
group_comparison_table_path = './group_comparison_table.png'
plt.savefig(group_comparison_table_path, bbox_inches='tight')

# Define the path for the CSV file
csv_file_path = './group_comparison_table.csv'

# Save the table to a CSV file
heatmap_with_groups_df.to_csv(csv_file_path, index=True)

# Define all behaviors
all_behaviors = list(data_clean.columns)

# Separate non-high-risk behaviors
non_high_risk_behaviors = [behavior for behavior in all_behaviors if behavior not in high_risk_behaviors]

# Create a DataFrame to display high-risk and non-high-risk behaviors
behaviors_df = pd.DataFrame({
    'High-Risk Behaviors': pd.Series(high_risk_behaviors),
    'Non-High-Risk Behaviors': pd.Series(non_high_risk_behaviors)
})

# Fill NaN values with empty strings
behaviors_df = behaviors_df.fillna('')

# Create a table figure
fig, ax = plt.subplots(figsize=(15, 10))  # Adjust height based on the number of behaviors
ax.axis('off')  # Hide axes

# Add the table to the figure
tbl = table(ax, behaviors_df, loc='center', cellLoc='left', colWidths=[0.4, 0.4])

# Customize table appearance
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.2, 1.2)  # Scale table for readability

# Save the table as an image
behavior_table_path = './high_risk_non_high_risk_behaviors_table.png'
plt.savefig(behavior_table_path, bbox_inches='tight')

# Define k value for KMeans
k = 6

# Get the cluster assignments for k=6
kmeans_clusters_k6 = cluster_results[k]['KMeans']

# Dictionary to hold table data for kmeans k=6 clusters
kmeans_k6_table_data = []

# Loop through each cluster to calculate the required statistics
for cluster in range(k):
    # Filter data for the current cluster
    cluster_data = data_clean[kmeans_clusters_k6 == cluster]
    
    # Calculate Yes/No counts and probabilities for each question
    yes_counts = cluster_data.sum().astype(int)
    no_counts = (len(cluster_data) - yes_counts).astype(int)
    yes_probabilities = ((yes_counts / len(cluster_data)) * 100).round(2)
    no_probabilities = ((no_counts / len(cluster_data)) * 100).round(2)
    
    # Store the results for this cluster
    cluster_results_df = pd.DataFrame({
        #f'C{cluster} Yes (n)': yes_counts.astype(int),
        #f'C{cluster} No (n)': no_counts.astype(int),
        f'C{cluster} Yes (%)': yes_probabilities,
        #f'C{cluster} No (%)': no_probabilities
    })
    
    # Append to the list
    kmeans_k6_table_data.append(cluster_results_df)

# Concatenate all cluster results into a single table for k=6
kmeans_k6_full_table = pd.concat(kmeans_k6_table_data, axis=1)

# Save the table to an image
fig, ax = plt.subplots(figsize=(4, 10))
ax.axis('off')

# Create the table
tbl = table(ax, kmeans_k6_full_table, loc='center', cellLoc='center', colWidths=[0.2] * kmeans_k6_full_table.shape[1])

# Customize table appearance
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1.1, 1.1)

# Save the table as an image
table_image_path = './kmeans_k6_cluster_table.png'
plt.savefig(table_image_path, bbox_inches='tight')

# Define the path for the CSV file
csv_file_path = './kmeans_k6_cluster_table.csv'

# Save the table to a CSV file
kmeans_k6_full_table.to_csv(csv_file_path, index=True)

# Separate the columns into high-risk and non-high-risk behaviors
high_risk_data = data_clean[high_risk_behaviors]
non_high_risk_data = data_clean.drop(columns=high_risk_behaviors)

# Count the number of "Yes" (1) responses for each data point in both high-risk and non-high-risk behaviors
high_risk_counts = high_risk_data.sum(axis=1)
non_high_risk_counts = non_high_risk_data.sum(axis=1)

# Combine these counts into a DataFrame and sort by non-high-risk counts in descending order
stacked_data = pd.DataFrame({'Non High Risk Yes': non_high_risk_counts, 'High Risk Yes': high_risk_counts})
stacked_data_sorted = stacked_data.sort_values(by='Non High Risk Yes', ascending=False).reset_index(drop=True)

# Plot the stacked bar chart
plt.figure(figsize=(14, 8))
plt.bar(stacked_data_sorted.index, stacked_data_sorted['Non High Risk Yes'], label='Non High Risk Yes', color='blue')
plt.bar(stacked_data_sorted.index, stacked_data_sorted['High Risk Yes'], bottom=stacked_data_sorted['Non High Risk Yes'], label='High Risk Yes', color='red')

# Add labels and legend
plt.xlabel('Data Points (Sorted by Non-High-Risk Count)')
plt.ylabel('Count of Yes Responses')
plt.title('Stacked Bar Chart of Non-High-Risk and High-Risk Yes Counts')
plt.legend()

# Show plot
plt.tight_layout()

# Save the plot as an image
stacked_bar_chart_path = './histogram_non_high_sort.png'
plt.savefig(stacked_bar_chart_path)

# Sort by high-risk counts in descending order
stacked_data_sorted_high_risk = stacked_data.sort_values(by='High Risk Yes', ascending=False).reset_index(drop=True)

# Plot the stacked bar chart, sorted by high-risk counts
plt.figure(figsize=(14, 8))
plt.bar(stacked_data_sorted_high_risk.index, stacked_data_sorted_high_risk['Non High Risk Yes'], label='Non High Risk Yes', color='blue')
plt.bar(stacked_data_sorted_high_risk.index, stacked_data_sorted_high_risk['High Risk Yes'], bottom=stacked_data_sorted_high_risk['Non High Risk Yes'], label='High Risk Yes', color='red')

# Add labels and legend
plt.xlabel('Data Points (Sorted by High-Risk Count)')
plt.ylabel('Count of Yes Responses')
plt.title('Stacked Bar Chart of Non-High-Risk and High-Risk Yes Counts (Sorted by High-Risk)')
plt.legend()

# Show plot
plt.tight_layout()

# Save the plot as an image
stacked_bar_chart_high_risk_sorted_path = './histogram_high_risk_sort.png'
plt.savefig(stacked_bar_chart_high_risk_sorted_path)
