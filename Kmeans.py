import os
import numpy as np
import matplotlib.pyplot as plt

from pdb import set_trace as breakpoint

# 参考： https://blog.csdn.net/orangefly0214/article/details/86538865

def findNN(centers, data, label, numCenters):
	for ind in range(data.shape[0]):
		curPoint = np.reshape(data[ind,:],(1,-1))
		diffSq = np.sum(np.square(centers - np.repeat(curPoint, numCenters, axis = 0)), axis = 1)
		indNN = np.argmin(diffSq)
		label[ind] = indNN;

	return label


def calcCenters(centers, label, data):
	newCenters = np.zeros_like(centers)
	for ind in range(centers.shape[0]):
		selected = data[label == ind]
		num = selected.shape[0]
		newCenters[ind] = np.sum(selected, axis = 0) / num
	return newCenters


def Kmeans(data, K):
	np.random.seed(0)
	centers = np.random.uniform(0,10,(K, data.shape[1]))
	label = np.zeros((data.shape[0]))

	print("start with centers: \n")
	print("center 1: (%.02f, %.02f)\n" %(centers[0][0], centers[0][1]))
	print("center 2: (%.02f, %.02f)\n" %(centers[1][0], centers[1][1]))
	print("center 3: (%.02f, %.02f)\n" %(centers[2][0], centers[2][1]))

	# curdat = np.reshape(data[0, :], (1, -1))
	# print(data[0,:], np.repeat(curdat, K, axis = 0).shape)
	cnt = 0
	label_Color = ['r','g','b']
	while True:

		# for ind in range(data.shape[0]):
		# 	curdata = np.reshape(data[ind, :], (1, -1))
		# 	diff = centers - np.repeat(curdata, K, axis = 0)
		# 	diffSq = np.sum(np.square(diff), axis = 1)
		# 	nnInd = np.argmin(diffSq)
		# 	print(diffSq, nnInd)
		# 	breakpoint()

		# 未知量 theta 是 cluster 的中心点，y 是 class label。
		label = findNN(centers, data, label, K)		# E step 求 p(y|theta)
		newCenters = calcCenters(centers, label, data)		# M step p(theta|y) , 根据各数据点分类，求新的中心位置。
		updateDiff = np.sqrt(np.sum(np.square(newCenters - centers)))
		centers = newCenters
		cnt += 1
		if (updateDiff < 1e-3):
			break

		plt.figure()
		for ind in range(3):
			curdata = data[label == ind]
			plt.scatter(centers[:,0], centers[:,1], marker= 'x', c= 'k')
			plt.scatter(curdata[:,0], curdata[:,1], c= label_Color[ind])
			plt.title('Iteration %d' %cnt)
		
	plt.show()
	return centers, cnt

if __name__ == '__main__':
	data1 = np.random.uniform(0,2, (10,2))
	data2 = np.random.uniform(3,6, (10,2))
	data3 = np.random.uniform(8,10, (10,2))
	# data = np.stack((data1, data2, data3), axis = 0)

	# print("start with centers: \n")
	# print("center 1: (%.02f, %.02f)\n", %(data1[0], data1[1]))
	# print("center 2: (%.02f, %.02f)\n", %(data2[0], data2[1]))
	# print("center 3: (%.02f, %.02f)\n", %(data3[0], data3[1]))

	data = np.concatenate((data1, data2, data3), axis = 0)
	np.random.shuffle(data)
	# print(data.shape)

	centers, cnt = Kmeans(data, 3)


	print("end with centers after %d interations: \n" % cnt)
	print("center 1: (%.02f, %.02f)\n" %(centers[0][0], centers[0][1]))
	print("center 2: (%.02f, %.02f)\n" %(centers[1][0], centers[1][1]))
	print("center 3: (%.02f, %.02f)\n" %(centers[2][0], centers[2][1]))

