# Course project in Deep Learning

# Title Project
"Using Deep Neural Network and Graph Convolutional Network to Estimate the Length of Shortest Path in Directed/Undirected Graphs for Primary Work"

# Introduction
Graphs are a fundamental data structure used to model relationships between entities in a variety of fields, including social networks, transportation systems, and biological networks. One important problem in graph theory is the estimation of the length of the shortest path between two nodes in a graph. This problem is particularly challenging in directed and undirected graphs, where the number of possible paths can be very large. In recent years, deep neural networks and graph convolutional networks have emerged as powerful tools for estimating the length of the shortest path in graphs. In this paper, I explore the challenges involved in estimating the length of the shortest path in directed and undirected graphs, and I investigate how deep neural networks and graph convolutional networks can be used to address these challenges. I examine the role of each of these techniques in the estimation process and present experimental results that demonstrate their effectiveness. My findings suggest that deep neural networks and graph convolutional networks can significantly improve the accuracy of shortest path estimation in graphs, and I argue that these techniques have important practical applications for tasks such as route planning and network analysis. 

# Data
The original datasets can be found here: https://www.cs.umd.edu/~sen/lbc-proj/LBC.html
This is citation for network data (Cora, Citeseer or Pubmed)
In this task we will use the Cora dataset from the LBP-Clustering
repository and perform a node classification on it using Graph Convolutional Networks (GCN).

# Methodology
We will implement two models to classify nodes of the graph: 1) Graph Convolutional
Networks with Spatial Attention Mechanism (GraphSAGE), and 2) Graph Con
volutional Neural Networks (GCN). 

# Requirements
* Python >=3.6
* PyTorch ==1.4.0
* *torch-cluster ==1.5.7
* torch-scatter ==2.0.5
* torch-spline-conv ==1.2.1
* tqdm>=4.38.0
* tensorboardX==2.1
* sklearn>=0.23.1
* matplotlib>=3.3.1
