from distancecomp_studentversion import pytorchdists
import numpy as np
import torch

def create_data(P, k, device):
    data = []
    labels = []
    for p in range(P): # iterates for the number of clusters we need
        mean = np.random.randn(2) # gives us the centre for the cluster
        cluster_data = torch.normal(mean=torch.tensor(mean, dtype=torch.float32).expand(k, 2), std=0.1) # generates k points by drawing from a normal distribution
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

if __name__ == '__main__':
    P = 8  # Antall klynger
    k = 50  # Punkter per klynge
    M = 10  # Maks iterasjoner
    device = torch.device('cuda')

    # Generer data
    X, Y = create_data(P, k, device)
    print("Data shape:", X.shape, Y.shape)

    centers, labels = k_means(X, P, M, device)
    print("Centers:", centers)
    print("Labels shape:", labels.shape)
