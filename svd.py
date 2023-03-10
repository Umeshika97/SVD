"""
Created on Thu Oct 28 09:34:55 2021
@author: umeshika
"""
#import the libraries
from datetime import datetime 
import numpy as np
import matplotlib.pyplot as plt
from skimage import  color
from scipy.linalg import svd
import seaborn as sns
import os
import math
#the starting time when the code was run
starttime=datetime.now()

#The function was defined to convert the data of matrix to row 
def matrix_to_row(A1):
    if  len(A1.shape) ==2:
        S=np.size(A1)
        B=np.reshape(A1,S)   #the rows was taken from matrix
        return B
    else:
        #print error
        print('error: The function support only for 2 dimentional matrices')
        return -1
        
#The function was defined to convert the row to the data of matrix 
def row_to_matrix(a,rows,columns):
    try:
        reshape = np.reshape(a,[rows,columns])#reshape the row to matrix
        return(reshape)
    except Exception as e:
        print(e)
        return(-1)


#%%
#load the database of image
image_path=('E:/03_Third Year/Second Semester/PH 3034 Digital Image Processing 1/week 06/specs/')
#E:/03_Third Year/Second Semester/PH 3034 Digital Image Processing 1/week 06/Cropped Images/
files=os.listdir(image_path)
#read the image 
image_data=plt.imread(image_path+files[0]);
#image dimension was taken
height,width,_=image_data.shape
#zero matrix was created
B=np.zeros([height*width])


#the images were taken in database
for f in files:
    original_img=plt.imread(image_path+f)
    #image was converted to gray image
    original_img=color.rgb2gray(original_img)
    #image data was converted to row
    original_img=matrix_to_row(original_img)
    #the row data stored 
    B=np.vstack([B,original_img])
    

B=B[1:,:]#skip the 0th row and take one row to the end

# %%
#Singular Value Decomposition was done
U,s,VT=svd(B,full_matrices=False)
print(U)
print(s)
print(VT)
#plot graphs with labels
plt.figure(1)
plt.plot(s,'*')
plt.title("The graph of Singular Values verse the number of index")
plt.xlabel("The number of index")
plt.ylabel("S matrix diagonal values")
plt.show()

plt.figure(2)
plt.semilogy(s,'*')
plt.title("The graph of S semilog values verse the number of index")
plt.xlabel("The number of index")
plt.ylabel("S semilog values")
plt.show()

plt.figure(3)
plt.plot(np.cumsum(np.diag(s))/np.sum(np.diag(s)))
plt.title('The graph of Singular Values Cumulative Sum')
plt.show()

# Variance were taken for Singular vectors
Variance_value = np.round(s**2/np.sum(s**2), decimals=6) 
print(Variance_value)
#bar graph was plot
sns.barplot(x=list(range(1, 7)),
            y=Variance_value[0:6], color="yellow")
plt.title('The Bar Graph of Variance')
plt.xlabel('The number of index')
plt.ylabel('Variance Value')
plt.tight_layout()
plt.show()
 
#%%image reconstruction and Root mean square error calculation
sum2=0
for i in range(U.shape[0]):
    for ii in files:
        truncation=4 #truncation vlue
        #image reconstruction with truncation
        Re_arrange=U[i,0:truncation] @np.diag(s[0:truncation]) @VT[0:truncation,:]
        final_image=row_to_matrix(Re_arrange,height,width) #final image
        plt.figure(i)
        plt.axis('off')#off the axis
        plt.imshow(final_image,cmap='gray')
        
        #crop the system image
        spoint1=round((final_image.shape[0])/4)
        spoint2=round((final_image.shape[0])*3/4)
        spoint3=round((final_image.shape[1])/4)
        spoint4=round((final_image.shape[1])*3/4)
        system_img=final_image[spoint1:spoint2,spoint3:spoint4]
        #plt.imshow(system_img,cmap='gray')

        #crop the original image   
        original_img=plt.imread(image_path+ii)
        ori_img=color.rgb2gray(original_img) 
        point1=round((ori_img.shape[0])/4)
        point2=round((ori_img.shape[0])*3/4)
        point3=round((ori_img.shape[1])/4)
        point4=round((ori_img.shape[1])*3/4)
        ori_img1=ori_img[point1:point2,point3:point4]
        #plt.figure(ii)
        #plt.axis('off')#off the axis
        #plt.imshow(ori_img,cmap='gray')
        #plt.imshow(ori_img1,cmap='gray')
        
        #error was calculated
        Error=ori_img1-system_img
        #plt.imshow(Error,cmap='gray')
        
        SE=(Error)**2 #error square value was taken
    #Root mean square error was calculated            
    sum2=0
    for j in range(SE.shape[0]):
        for g in range(SE.shape[1]):
            sum2=sum2+SE[j,g]
    #print the RMSE     
    #print(files[i],'square error is ',sum2)        
    RMSE=math.sqrt(sum2)       
    print(files[i],'Root mean square error is ',RMSE)
    
#the code execution time was taken
print (datetime.now()-starttime)        

