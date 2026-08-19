import numpy as np
import torch
import os
import glob

from torch.utils.data import Dataset
from torch.nn import functional as F
import torchvision.transforms as transforms
import shapeworks as sw

from sklearn.mixture import GaussianMixture
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation

class Sampler(ABC):
	# abstract methods
	# fit should fit a distribution to the embedded_matrix
	def fit(self, embedded_matrix):
		self.embedded_matrix = embedded_matrix
		pass
	# sample should return the sample and the index of the nearest real example
	def sample(self):
		pass

class Gaussian_Sampler(Sampler):
	def fit(self, embedded_matrix):
		# print("Fitting Gaussian distribution...")
		self.embedded_matrix = embedded_matrix
		self.mean = np.mean(embedded_matrix, axis=0)
		self.cov = np.cov(embedded_matrix, rowvar=0)
		if self.embedded_matrix.shape[1] == 1:
			self.cov = np.array([[self.cov]])
	def sample(self):
		sample = np.random.multivariate_normal(self.mean, self.cov)
		closest_index = getClosest(sample, self.embedded_matrix)
		return sample, closest_index

class Mixture_Sampler(Sampler):
	def fit(self, embedded_matrix, mixture_num):
		print("Fitting Gaussian mixture model...")
		self.embedded_matrix = embedded_matrix
		if mixture_num == 0:
			mixture_num = self.selectClusterNum()
		self.GMM = GaussianMixture(mixture_num, covariance_type='full', random_state=0)
		self.GMM.fit(self.embedded_matrix)
		print("Gaussian mixture model converged: " + str(self.GMM.converged_))
	# get optimal cluster number by minimizing Akaike information criterion (AIC) and Bayesian information criterion (BIC)
	def selectClusterNum(self):
		n_components = np.arange(1, self.embedded_matrix.shape[1])
		models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(self.embedded_matrix) for n in n_components]
		bic_min_index = np.argmin(np.array([m.bic(self.embedded_matrix) for m in models]))
		aic_min_index = np.argmin(np.array([m.aic(self.embedded_matrix) for m in models]))
		avg_index = int((bic_min_index + aic_min_index) / 2)
		mixture_num = n_components[avg_index]
		print("Using " + str(mixture_num) + " components.")
		return mixture_num
	def sample(self):
		sample = self.GMM.sample(1)[0][0]
		closest_index = getClosest(sample, self.embedded_matrix)
		return sample, closest_index

# instance of Sampler class that uses kernel density estimate
class KDE_Sampler(Sampler):
	def fit(self, embedded_matrix):
		print("Fitting KDE...")
		self.embedded_matrix = embedded_matrix
		# get sigma squared
		nearest_neighbor_dists = []
		cov = np.cov(embedded_matrix.T)
		for i in embedded_matrix:
			smallest = np.Inf
			for j in embedded_matrix:
				dist = Mdist(i,j,cov)
				if dist < smallest and dist != 0:
					smallest = dist
			nearest_neighbor_dists.append(smallest)
		self.sigma_squared = np.mean(np.array(nearest_neighbor_dists))/embedded_matrix.shape[1]
	def sample(self):
		base_index = np.random.randint(self.embedded_matrix.shape[0])
		base_PCA_score = self.embedded_matrix[base_index]
		noise = []
		for i in range(self.embedded_matrix.shape[1]):
			noise.append(np.random.normal(0,self.sigma_squared))
		noise = np.array(noise)
		sampled_PCA_score = base_PCA_score + noise
		return sampled_PCA_score, base_index

# sampler helper - gets mahalanobis distance
def Mdist(instance_i, instance_j, covariance_matrix):
	temp = instance_i - instance_j
	dist = np.dot(np.dot(temp, np.linalg.inv(covariance_matrix)), temp.T)
	return dist

# sampler helper - gets closest real example to sample
def getClosest(sample, embedded_matrix):
	covariance_matrix = np.cov(embedded_matrix.T)
	if embedded_matrix.shape[1] == 1:
		covariance_matrix = np.array([[covariance_matrix]])
	smallest = np.inf
	for index in range(embedded_matrix.shape[0]):
		example = embedded_matrix[index]
		dist = Mdist(sample, example, covariance_matrix)
		if dist <= smallest:
			smallest = dist
			closest = index
	return closest




def contruct_translation_sampler(property_files):
	R_list = []
	t_list = []
	s_list = []

	for i in range(len(property_files)):
		rigid_transform = np.load(property_files[i])["rigid_transform"]
		R, t, s = similarity_homogeneous_get_rtc(rigid_transform)
		R_list.append(R)
		t_list.append(t)
		s_list.append(s)
	
	tSampler = Gaussian_Sampler()
	tSampler.fit(np.array(t_list)) 
	tSampler.cov = tSampler.cov

	RSampler = Gaussian_Sampler()
	RSampler.fit(np.array(R_list)) 
	RSampler.cov = RSampler.cov

	sSampler = Gaussian_Sampler()
	sSampler.fit(np.array(s_list)) 
	sSampler.cov = sSampler.cov

	return  RSampler, tSampler, sSampler


def contruct_Rts_sampler(data_dict, label):
	mat_list = []

	for cd in data_dict:
		if len(cd["segmentations"][label]) > 0:
			rigid_transform = np.load(cd["segmentations"][label].replace("segmentations", "properties").replace(".nrrd", ".npz"))["rigid_transform"]
			R, t, s = similarity_homogeneous_get_rtc(rigid_transform)
			trs = np.concatenate([R, t.reshape((-1)), np.array([s])])
			mat_list.append(trs)
	

	RtsSampler = KDE_Sampler()
	RtsSampler.fit(np.array(mat_list)) 
	# RtsSampler.cov = RtsSampler.cov

	return  RtsSampler



def similarity_homogeneous_get_rtc(homo_matrix):

	t = homo_matrix[:3, 3]

	c = np.sum(np.sum(homo_matrix[:3, :3], axis=1)**2)**0.5 / (3**0.5)

	R = homo_matrix[:3,:3] / c

	R = Rotation.from_matrix(R).as_rotvec()

	return R, t, c


def get_homogeneous_similar_transform(R, t, c):

	m, m = R.shape
	T = np.zeros((m+1, m+1))
	T[0:m, 0:m] = c * R
	T[0:m, [3]] = t
	T[m,m] = 1
	return T