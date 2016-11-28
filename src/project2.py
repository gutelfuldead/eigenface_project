#!/usr/bin/python2.7
import math
import numpy as np
import matplotlib.pyplot as plt
from proj2_functions import pca, kNN, norm_vectors, calc_eigface_var, data_projection, retrieve_data
import re
from timeit import default_timer as timer

# To suppress outputs change to False
PART1 = True
# PART1 = False

PART2 = True
# PART2 = False

PART3 = True
# PART3 = False

PART4a5 = True
# PART4a5 = False

showplots = True
# showplots = False

# saveplots = True
saveplots = False

#########################################
# Import probeset and galleryset images
#########################################

probe_set, probe_labels = retrieve_data('../data/ProbeSet')
gallery_set, gallery_labels = retrieve_data('../data/GallerySet')

##############################################################################################
# part 1
# Using the gallery set to compute the PCA projection of the data, display the first three
# principal components as face images. We know that the first three components represent
# the directions of highest variance in the data but what does the largest principal component
# represent in terms of facial recognition?
##############################################################################################
if PART1 == True:
	[e,EV,mu] = pca(gallery_set)
	m=n=50
	plt.figure()
	plt.gray()
	plt.subplot(2,2,1)
	plt.imshow(mu.reshape(m,n))
	plt.title("Mean face")
	plt.subplot(2,2,2)
	plt.imshow(EV[:,0].reshape(m,n))
	plt.title("1st Eigenface")
	plt.subplot(2,2,3)
	plt.imshow(EV[:,1].reshape(m,n))
	plt.title("2nd Eigenface")
	plt.subplot(2,2,4)
	plt.imshow(EV[:,2].reshape(m,n))
	plt.title("3rd Eigenface")
	if saveplots == True:
		plt.savefig('../images/1_eigenfaces.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla


##############################################################################################
# part 2
# Using the eigen-coefficients, Euclidean distance as the distance measure, and varying the
# number of principal components from 10 to 100 in steps of 10, plot the recognition rate of
# as a function of the number of principal components used. Discuss the observed trend in
# recognition performance. Would you expect this trend to continue as the number of principal
# components increases? Explain your reasoning.
##############################################################################################
if PART2 == True:
	[e,EV,mu] = pca(gallery_set)
	EV = norm_vectors(EV)
	err = np.zeros(10)

	# old = 0
	for i in range(0,101):
			a = calc_eigface_var(e,i/100.0)
			# if a > old or i == 100:
				# print("{} prinicpal components required to account for {}% of the total variance").format(old,i)
			print("{} prinicpal components required to account for {}% of the total variance").format(a,i)
			# old = a
	print "\n"

	for i in range(1,11):
		err[i-1],_ = kNN(gallery_set, probe_set, mu, EV, gallery_labels, probe_labels, numPCA=10*i)

	plt.figure()
	plt.plot(np.linspace(10,100,10),err)
	plt.title("Recognition Rate v. Number of Eigenfaces")
	plt.xlabel("Number of Eigenfaces used")
	plt.ylabel("Recognition rate in %")
	if saveplots == True:
		plt.savefig('../images/2_recognition_rate_PCA.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

	##############################################################
	# BONUS
	# Drop first eigenvector as it only contains light information
	##############################################################
	print ("\nFirst eigenvector dropped to improve performance...")
	for i in range(1,11):
		err[i-1],ttime = kNN(gallery_set, probe_set, mu, EV[:,1::], gallery_labels, probe_labels, numPCA=10*i)

	plt.figure()
	plt.plot(np.linspace(10,100,10),err)
	plt.title("Recognition Rate v. Number of Eigenfaces")
	plt.xlabel("Number of Eigenfaces used")
	plt.ylabel("Recognition rate in %")
	if saveplots == True:
		plt.savefig('../images/2_recognition_rate_PCA_first_comp_removed.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

##############################################################################################
# part 3
# Using Euclidean distance as the distance measure and the original images as feature vectors,
# determine recognition performance. How does the performance obtained compare with that
# obtained using the PCA projection of the data?
##############################################################################################
if PART3 == True:
	from scipy.spatial.distance import euclidean as dist
	# try to match a gallery image
	correct = 0
	start = timer()
	for i in range (0,len(probe_labels)):
		err_k = 100000
		match = 99999
		for j in range (0,len(gallery_labels)):
			if dist(probe_set[i],gallery_set[j]) < err_k:
				err_k = dist(probe_set[i],gallery_set[j])
				match = j
				# print i,j
		if gallery_labels[match] == probe_labels[i]:
			# print match,i
			correct += 1
	end = timer()
	print ("\nUsing original images as feature vectors...")
	print "Percent correct using original image = {:.2f}%".format(float(correct)/len(probe_labels)*100)

	# show size comparison
	print "Original feature total gallery size = {}, classification time = {}s".format(gallery_set.size,end-start)
	print "Feature reduced total gallery size = {}, classification time = {}s".format(\
		data_projection(din=gallery_set,numPCA=100,EV=EV[:,1::],mu=mu).size, ttime)

######################################################################################################
######################################################################################################
# When performing large-scale facial recognition, a way to improve performance is to reduce the
# search space for matches by first performing soft biometric classification (gender, ethnicity, age).
######################################################################################################
######################################################################################################

###############################################################################################
# parts 4 and 5
# Using the eigen-coefficients, Euclidean distance as the distance measure, and varying the
# number of principal components from 10 to 100 in steps of 10, perform K-means clustering
# where K = 2. Each cluster should be composed of images from individuals of a single gender.
#
# Evaluate the clustering in 1 using an internal criteria and an external criteria. Plot cluster
# validity as a function of the number principal components. Are the observed trends compa-
# rable to those observed in terms of recognition? Would you expect these trends to continue
# as the number of principal components increases? Explain your reasoning.
###############################################################################################
if PART4a5 == True:
	# Gender.txt contains list of male/female for each person
	import pandas as pd
	from sklearn.cluster import KMeans
	from scipy.cluster.vq import kmeans2 as km2
	from sklearn.metrics import f1_score
	from jqmcvi.base import dunn_fast, davisbouldin
	import matplotlib.patches as mpatches

	gender_file = '../data/Gender.txt'
	df = pd.read_csv(gender_file)
	saved_column = df['GENDER']
	# convert to binary vector 1 == male 0 == female
	gend_bin = np.zeros(len(saved_column),dtype=np.int8)
	for i in range(0,len(saved_column)):
		if saved_column[i] == 'male':
			gend_bin[i] = 1
		else:
			gend_bin[i] = 0

	[e,EV,mu] = pca(gallery_set)
	EV = norm_vectors(EV)

	F1 = np.zeros(10)
	dn = np.zeros(10)
	accuracy = np.zeros(10)

	red = mpatches.Patch(color='r',  label='Male')
	blue = mpatches.Patch(color='b',label='Female')

	print("\nProvide kmeans centroids of averaged male and female gallery data based on number of eigenfaces kept")
	for i in range(1,11):
		project_gallery = data_projection(din=gallery_set,numPCA=10*i,EV=EV[:,1::],mu=mu)
		project_probe   = data_projection(din=probe_set,numPCA=10*i,EV=EV[:,1::],mu=mu)

		# must reindex each time due to changing projections...
		male = []
		female = []
		for j in range(0,len(project_gallery)):
			if gend_bin[gallery_labels[j]-1] == 1:
				male.append(project_gallery[j])
			else:
				female.append(project_gallery[j])
		male = np.asarray(male)
		female = np.asarray(female)

		# generate means for each
		m_mu = male.mean(axis=0)
		f_mu = female.mean(axis=0)
		initial_centroids = np.array([m_mu,f_mu])

		# compute k means
		cluster_centroids, kmeans_labels = km2(data=project_probe,k=initial_centroids, minit = 'matrix')
		accuracy[i-1] = float(np.sum(gend_bin == kmeans_labels))/len(gend_bin)*100.0

		# calc dunn and F1 index for k = 2 to 10
		dn[i-1] = dunn_fast(points=project_probe, labels=kmeans_labels)
		F1[i-1] = f1_score(y_true=gend_bin,y_pred=kmeans_labels)

		print "numPCA = {}; Dunn = {:.5f}; F1 = {:.5f}; percent correct = {:.2f}".format(i*10,dn[i-1],F1[i-1],accuracy[i-1])

		if i == 1 or i == 10:
			plt.figure()
			plt.scatter(project_probe[:,0],project_probe[:,1],project_probe[:,2],c=kmeans_labels.astype(np.float))
			plt.xlabel("First Principal Component")
			plt.ylabel("Second Principal Component")
			ttl = "Clustering based on first two PC with total of {} PCs".format(i*10)
			plt.title(ttl)
			plt.legend(handles=[red,blue])
			if saveplots == True:
				plt.savefig('../images/4_MF_sep_clustering_with_{}_PCs.png'.format(i*10))
			if showplots == True:
				plt.show()
			plt.clf
			plt.cla

			plt.figure()
			plt.plot(range(0,len(cluster_centroids[0])),cluster_centroids[0],label="Cluster 0")
			plt.plot(range(0,len(cluster_centroids[1])),cluster_centroids[1],label="Cluster 1")
			plt.xlabel("Dimension")
			plt.ylabel("Dimension Centroid")
			ttl = "Number of Dimensions v. Centroids for {} PCs".format(i*10)
			plt.title(ttl)
			plt.legend()
			if saveplots == True:
				plt.savefig('../images/4_MF_sep_dim_v_centroids_{}_PCs.png'.format(i*10))
			if showplots == True:
				plt.show()
			plt.clf
			plt.cla

	# plot dunn index
	plt.figure()
	plt.plot(np.linspace(10,100,10),dn)
	plt.xlabel("Number of Primary Components")
	plt.ylabel("Dunn Index Value")
	plt.title("Dunn Index v. Number of Primary Components")
	if saveplots == True:
		plt.savefig('../images/4_MF_sep_dunn_index.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

	# plot F1 index
	plt.figure()
	plt.plot(np.linspace(10,100,10),F1)
	plt.xlabel("Number of Primary Components")
	plt.ylabel("F1 Index Value")
	plt.title("F1 Index v. Number of Primary Components")
	if saveplots == True:
		plt.savefig('../images/4_MF_sep_F1_index.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

	# plot total accuracy
	plt.figure()
	plt.plot(np.linspace(10,100,10),accuracy)
	plt.xlabel("Number of Primary Components")
	plt.ylabel("Accuracy in percentage")
	plt.title("Accuracy v. Number of Primary Components")
	if saveplots == True:
		plt.savefig('../images/4_MF_sep_accuracy')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

	print("\nGenerate centroids naively from kmeans algorithm using gallery set before applying to probe set")
	for i in range(1,11):
		# calculate k means
		project_gallery = data_projection(din=gallery_set,numPCA=10*i,EV=EV[:,1::],mu=mu)
		project_probe   = data_projection(din=probe_set,numPCA=10*i,EV=EV[:,1::],mu=mu)
		kmeans = KMeans(n_clusters=2)
		kmeans.fit(project_gallery)
		lbl = kmeans.predict(project_probe)
		accuracy[i-1] = float(np.sum(gend_bin == lbl))/len(project_probe)*100.0

		# calc dunn and F1 index for k = 2 to 10
		dn[i-1] = dunn_fast(points=project_probe, labels=lbl)
		F1[i-1] = f1_score(y_true=gend_bin,y_pred=lbl)

		print "numPCA = {}; Dunn = {:.5f}; F1 = {:.5f}; percent correct = {:.2f}".format(i*10,dn[i-1],F1[i-1],accuracy[i-1])

		if i == 1 or i == 10:
			plt.figure()
			plt.scatter(project_probe[:,0],project_probe[:,1],project_probe[:,2],c=lbl.astype(np.float))
			plt.xlabel("First Principal Component")
			plt.ylabel("Second Principal Component")
			ttl = "Clustering based on first two PC with total of {} PCs".format(i*10)
			plt.title(ttl)
			plt.legend(handles=[red,blue])
			if saveplots == True:
				plt.savefig('../images/4_adapted_weights_clustering_with_{}_PCs.png'.format(i*10))
			if showplots == True:
				plt.show()
			plt.clf
			plt.cla

			plt.figure()
			plt.plot(range(0,len(kmeans.cluster_centers_[0])),kmeans.cluster_centers_[0],label="Cluster 0")
			plt.plot(range(0,len(kmeans.cluster_centers_[1])),kmeans.cluster_centers_[1],label="Cluster 1")
			plt.xlabel("Dimension")
			plt.ylabel("Dimension Centroid")
			ttl = "Number of Dimensions v. Centroids for {} PCs".format(i*10)
			plt.title(ttl)
			plt.legend()
			if saveplots == True:
				plt.savefig('../images/4_adapted_weights_dim_v_centroids_{}_PCs.png'.format(i*10))
			if showplots == True:
				plt.show()
			plt.clf
			plt.cla

	# plot dunn index
	plt.figure()
	plt.plot(np.linspace(10,100,10),dn)
	plt.xlabel("Number of Primary Components")
	plt.ylabel("Dunn Index Value")
	plt.title("Dunn Index v. Number of Primary Components")
	if saveplots == True:
		plt.savefig('../images/4_adapted_weights_dunn_index.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

	# plot F1 index
	plt.figure()
	plt.plot(np.linspace(10,100,10),F1)
	plt.xlabel("Number of Primary Components")
	plt.ylabel("F1 Index Value")
	plt.title("F1 Index v. Number of Primary Components")
	if saveplots == True:
		plt.savefig('../images/4_adapted_weights_index.png')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla

	# plot total accuracy
	plt.figure()
	plt.plot(np.linspace(10,100,10),accuracy)
	plt.xlabel("Number of Primary Components")
	plt.ylabel("Accuracy in percentage")
	plt.title("Accuracy v. Number of Primary Components")
	if saveplots == True:
		plt.savefig('../images/4_adapted_weights_accuracy')
	if showplots == True:
		plt.show()
	plt.clf
	plt.cla
