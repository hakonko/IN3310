from distancecomp_studentversion import pytorchdists
import numpy as np
import torch
import matplotlib.pyplot as plt

def create_data(P, k, device):
    data = []
    labels = []
    for p in range(P): # iterates for the number of clusters we need
        mean = np.random.uniform(-1, 1, size=2) # gives us the centre for the cluster
        cluster_data = torch.normal(mean=torch.tensor(mean, dtype=torch.float32).expand(k, 2), std=0.5) # generates k points by drawing from a normal distribution
        data.append(cluster_data)
        labels.append(torch.full((k,), p)) # creates a long tensor with the labels for each cluster

    return torch.cat(data, dim=0).to(device), torch.cat(labels, dim=0).to(device) # concatenates down to tensors of size (P*k, 2)

def k_means(X, P, M, device):
    # X = data points (N, 2)
    # P = number of clusters
    # M = max number of iterations

    indices = torch.randperm(X.shape[0])[:P] # randomizes the indices and chooses P of them, for our cluster centers
    centers = X[indices].clone().to(device) # creates this and sends to the device

    for m in range(M):
        distances = pytorchdists(X, centers, device) # uses pytorchdist to calculate the L2-distances
        labels = torch.argmin(distances, dim=1) # finds the nearest cluster for the data point

        new_centers = torch.zeros_like(centers, device=device) # creates a placeholder for new center points
        for p in range(P):
            cluster_points = X[labels == p] # finds the points that belong to each cluster
            if len(cluster_points) > 0: # checks that we actually have points in the cluster
                new_centers[p] = cluster_points.mean(dim=0) # updates new_centers with the means of the points
                                                     
        if torch.allclose(new_centers, centers, atol=1e-6): # stop criterion: if there has not been a change bigger than 1e-6
            break

        centers = new_centers # finally updating centers (means)
    
    return centers, labels

def plot_clusters(X, labels, centers, filename):
    X_np = X.cpu().numpy() # moving to cpu again and converting to numpy for plotting
    labels_np = labels.cpu().numpy() # likewise with these
    centers_np = centers.cpu().numpy()

    plt.figure(figsize=(8, 6))

    # plotting the points with color based on cluster
    plt.scatter(X_np[:, 0], X_np[:, 1], c=labels_np, cmap='tab10', s=10, alpha=0.6)
    
    # plotting cluster centers
    plt.scatter(centers_np[:, 0], centers_np[:, 1], c='red', marker='x', s=100, label='Cluster Centers')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('k-Means Clustering')
    plt.legend()

    # saving figure
    plt.savefig(filename, dpi=300)
    print(f"Plot saved as {filename}")
    plt.close()

if __name__ == '__main__':
    P = 8  # clusters
    k = 50  # points per cluster
    M = 10  # max iterations
    device = torch.device('cuda')
    filename = 'k_means.png'

    # creating data
    X, Y = create_data(P, k, device)
    print("Data shape:", X.shape, Y.shape)

    centers, labels = k_means(X, P, M, device)
    print("Centers:", centers)
    print("Labels shape:", labels.shape)

    plot_clusters(X, labels, centers, filename)
