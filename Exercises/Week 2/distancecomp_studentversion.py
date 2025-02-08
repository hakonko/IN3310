import os,sys,numpy as np
import torch
import time


def forloopdists(feats,protos):
  dists = np.zeros((feats.shape[0], protos.shape[0]))
  for i in range(feats.shape[0]):
    for j in range(protos.shape[0]):
      dist = 0
      for k in range(feats.shape[1]):
        dist += (feats[i, k] - protos[j, k])**2 # L2-distance
      dists[i, j] = np.sqrt(dist)
  
  return dists

# using that ||X - Y||^2 is X^2 + Y^2 - 2XY^T

def numpydists(feats,protos):
  X_sq = np.sum(feats**2, axis=1, keepdims=True)
  Y_sq = np.sum(protos**2, axis=1, keepdims=True)
  crossterm = 2 * np.matmul(feats, protos.T)

  dists = np.sqrt(X_sq + Y_sq.T - crossterm)
  return dists
  
def pytorchdists(feats0,protos0,device):
  feats = torch.tensor(feats0, dtype=torch.float32, device=device)
  protos = torch.tensor(protos0, dtype=torch.float32, device=device)

  X_sq = torch.sum(feats**2, dim=1, keepdim=True)
  Y_sq = torch.sum(protos**2, dim=1, keepdim=True)
  crossterm = 2 * torch.matmul(feats, protos.T)

  dists = torch.sqrt(X_sq + Y_sq.T - crossterm)
  return dists
  
def run():

  ########
  ##
  ## if you have less than 8 gbyte, then reduce from 250k
  ##
  ###############
  line = '='*30
  feats=np.random.normal(size=(5000,300)) #5000 instead of 250k for forloopdists
  protos=np.random.normal(size=(500,300))

  # Naive looping (NB! will take in excess of 300 secs!)
  since = time.time()
  dists0=forloopdists(feats,protos)
  forloopdists(feats, protos)
  time_elapsed=float(time.time()) - float(since)
  print(line, '\nNaive looping: Computation complete in {:.3f}s'.format( time_elapsed ))
  print(dists0.shape)


  feats=np.random.normal(size=(250000,300)) #changing to a bigger feats-matrix

  # Numpy-implementation
  since = time.time()
  dists2=numpydists(feats,protos)
  time_elapsed=float(time.time()) - float(since)
  print(line, '\nNumpy: Computation complete in {:.3f}s'.format( time_elapsed ))
  print(dists2.shape)

  # Torch-implementation with CPU and GPU (Cuda) usage
  devices = [torch.device('cpu'), torch.device('cuda')]

  for device in devices:
    since = time.time()
    dist=pytorchdists(feats,protos,device)
    time_elapsed=float(time.time()) - float(since)
    print(line, f'\nTorch with {device.type}: Computation complete in {time_elapsed:.3f}s')
    print(dist.shape)

if __name__=='__main__':
  run()