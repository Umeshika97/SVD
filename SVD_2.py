# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:02:14 2021

@author: Milash Heyn
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import color
from scipy import linalg
#from skimage import io
def matrix_to_row(A1):
    if  len(A1.shape) ==2:
        S=np.size(A1)
        #print(S) 
        B=np.reshape(A1,S)   #Taking the rows as a line
        return B
    else:
        print('error: The function support only for 2 dimentional matrices')
        return -1
    
image_path = ('E:/03_Third Year/Second Semester/PH 3034 Digital Image Processing 1/week 06/Cropped Images/')
#E:\03_Third Year\Second Semester\PH 3034 Digital Image Processing 1\week 06\Cropped Images

files = os.listdir(image_path)
b = plt.imread(image_path+files[0])
len1, len2,_ = b.shape
b = color.rgb2gray(b)
b = matrix_to_row(b)
D = np.zeros([1, b.shape[0]])
#print(D)
for f in files:
    a = plt.imread(image_path+f)
    a = color.rgb2gray(a)
    a = matrix_to_row(a)
    D = np.vstack([D,a])

D = D[1:,:]
#print(D)

def row_to_matrix(a,rows,columns):
    try:
        reshape = np.reshape(a,[rows,columns])
        return(reshape)
        print(reshape)
    except Exception as e:
        print(e)
        return(-1)
    
t = D[2,:]
graph = row_to_matrix(t,len1, len2)  
plt.imshow(graph, cmap='gray')

[u, s, vT] = np.linalg.svd(D, full_matrices=False)

plt.semilogy(s, '*')

j = row_to_matrix(vT[0:1,:], len1, len2)
plt.imshow(j, cmap='gray')


trunc = 10
re_a = u[16, 0:trunc] @ np.diag(s[0:trunc]) @ vT[0:trunc, :] 
#re_a = np.matmul(u[1,0:trunc], vT[0:trunc, :])
#plt.imshow(re_a, cmap='gray')

re_a2 = row_to_matrix(re_a, len1, len2)
plt.imshow(re_a2, cmap='gray')







