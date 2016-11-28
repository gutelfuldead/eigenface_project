#!/bin/bash
# proj2 functions
import matplotlib.pyplot as plt
import glob
import re
import numpy as np
import os

def read_pgm(filename, byteorder='>'):
	"""Return image data from a raw PGM file as numpy array.

	Format specification: http://netpbm.sourceforge.net/doc/pgm.html

	ref : http://stackoverflow.com/questions/7368739/numpy-and-16-bit-pgm/7369986#7369986
	"""
	with open(filename, 'rb') as f:
		buffer = f.read()
	try:
		header, width, height, maxval = re.search(
			b"(^P5\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	return np.frombuffer(buffer,
							dtype='u1' if int(maxval) < 256 else byteorder+'u2',
							count=int(width)*int(height),
							offset=len(header)
							).reshape((int(height)*int(width)))

def retrieve_data(DIR):
	tmp = []
	tmp2 = []
	d_set = np.zeros((len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]),2500))
	d_lbl = np.zeros(len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))]))
	for filename in sorted(glob.iglob(DIR+'/*.pgm')):
		tmp.append(read_pgm(filename))
		a = (re.findall(r'\d+', filename[23:30])) # find the person ID
		tmp2.append(int(a[0]))
	for i in range(0,len(d_set)):
		d_set[i,:] = np.asarray(tmp[i])
	d_labels = np.asarray(tmp2)
	return d_set,d_labels

def pca(X):
	'''
	ref : https://github.com/bytefish/facerecognition_guide/blob/master/src/py/tinyfacerec/subspace.py
	'''
	[n,d] = X.shape
	mu = X.mean(axis=0)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else:
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	# or simply perform an economy size decomposition
	# eigenvectors, eigenvalues, variance = np.linalg.svd(X.T, full_matrices=False)
	# sort eigenvectors descending by their eigenvalue
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	# select only num_components
	eigenvalues = eigenvalues[0:n].copy()
	eigenvectors = eigenvectors[:,0:n].copy()
	return [eigenvalues, eigenvectors, mu]

def calc_eigface_var(e,var):
	'''
	calculate how many of the eigenvectors are needed in order to account for VAR
	of all the possible variation
	input:
		e : np array
			array of eigenvalues
		var : float
			0 < var < 1 the amount of variation to account for
	returns:

	'''
	eigsum = sum(e)
	csum = 0.0
	tv = 0.0
	for i in range(0,len(e)):
		csum += e[i]
		tv = csum/eigsum
		if tv > var:
			# print("{} prinicpal components required to account for {}% of the total variance").format(i,var*100)
			return i

import math
def norm_vectors(eigenvectors):
	'''
	ref : https://github.com/Sebac/Eigenfaces_FaceRecognition/blob/master/face_recognition.py
	'''

	eigenvectors = eigenvectors.transpose()
	res = []

	for i in range(eigenvectors.shape[0]):
		suma = math.sqrt(sum([x**2 for x in eigenvectors[i]]))
		res.append([x/suma for x in eigenvectors[i]])
		# print res[i]

	return np.array(res).transpose()

def data_projection(din,numPCA,EV,mu):
	a = np.zeros((len(din),numPCA))
	for i in range(0,len(a)):
		a[i] = np.transpose(EV[:,0:numPCA]).dot(din[i]-mu)
	return a

from scipy.spatial.distance import euclidean as dist
from timeit import default_timer as timer
def kNN(gallery_set, probe_set, meanface, eigenfaces, gallery_labels, probe_labels, numPCA):
	'''
	im_in : np.array((M,M))
		input image
	mean_faces : np.array
		average face from training data
	eigen_faces : np.array
		eigenfaces from training data
	numPCA : int
	'''

	Omega_in = data_projection(probe_set,numPCA,eigenfaces,meanface)
	Omega_tr = data_projection(gallery_set,numPCA,eigenfaces,meanface)

	# try to match a gallery image
	correct = 0
	start = timer()
	for i in range (0,len(probe_labels)):
		err_k = 100000
		match = 99999
		for j in range (0,len(gallery_labels)):
			if dist(Omega_in[i],Omega_tr[j]) < err_k:
				err_k = dist(Omega_in[i],Omega_tr[j])
				# print dist(Omega_in[i],Omega_tr[j])
				# print gallery_labels[j],probe_labels[i]
				match = j
		if gallery_labels[match] == probe_labels[i]:
			correct += 1
	end = timer()

	print "Prinicipal Components = {}; ttl time = {:.5f}s; Percent correct = {:.2f}%".format(numPCA,end-start,float(correct)/len(probe_labels)*100)
	return float(correct)/len(probe_labels)*100, end-start
