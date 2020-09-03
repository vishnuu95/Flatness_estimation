import numpy as numpy
import scipy 
import matplotlib.pyplot as plt
import cv2
import argparse

def gmm_predict():
	pass

def gmm_update():	
	pass

def gmm_init(n_d=3, n_g = 3, n_m = np.array([100,100,100])):
	np.random.seed(1)
	model = {}
	model["mean"] = np.zeros((n_d, 3))
	model["var"] = np.zeros((n_d, 3, 3))
	for i in range (n_d):
		model["mean"][i,:] = n_mean + np.random.randint(50, size=(1,3))
		cov = np.random.randint(low = 1, high=30, size=(3,3))
		model["var"][i,:] = (cov + cov.T)/2
	return model	

def save_model():
	pass

if __init__=="__main__":
	parser = argparse.ArgumentParser(description="options to give input and ouput folder")
	n_dims = 3
	n_gaussians = 3
	n_mean = np.array([100,100,100])		

	# Initialise the model
	model = gmm_init(n_dims = n_dims, n_gaussians= n_gaussians, n_mean= n_mean)
	print (model)
	parser.add_argument('--input', default='./dataset' )
	parser.add_argument('--output', default = './model_output')

	print(output)




