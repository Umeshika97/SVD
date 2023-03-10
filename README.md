# SVD
(Singular Value Decomposition)

# 1 INTRODUCTION 

In the present world, innovation is revolute and supplanted by another technology generation by
generation. Thusly, medical diagnosis imaging, industrial work, agriculture, educational field, and 
space research field are also a revolution day by day because of new inventions. Consequently,
digital image processing is needed to do the image mining (image compression) that is described
as the data mining technique where images are primarily utilized as data. Therefore, RGB images 
can be considered as the lattice with pixels addressed as individual mathematical formulas. This 
practical mainly considers the 2-dimensional 6 gray images with two databases which can be 
obtained as the 2-d array matrix which is included an immersion size of 0 to 255 for every pixel.
In addition, the fundamental data mining technique of image processing is the Singular Value 
Decomposition which can be precisely compressed the image data with less mean square error.
In Singular Value Decomposition (SVD) process includes separating through a matrix into the 
structured form. This conversion is permitted to hold the significant solitary qualities that the 
image requires while likewise delivering the qualities that are not as essential in holding the 
excellence of the image. The resultant image of this image data mining technique has removed
concealed data and which is linked to image data and further patterns which are extremely not 
visible in the image. The main advantage of utilizing this technique is Singular Value Decay to pack 
the size of the inundation groups, it is keeping the unique data, and to store an image using less 
memory while in a like approach keeping the image quality
Subsequently, the primary purpose behind this practical was to Implement and visualize the 
compressed image with a less mean square error by utilizing the open programming of the 
excellent level programming language of Python 03. Python 03 is an article arranged prearranging 
language, and it has been simple to consequently the amazing benefit of learning python is that 
has allowed troubleshooting of bits of code and intuitive testing. Present, python 03 is used in a 
lot of fields of advanced images handling the way that everyone can download straightforwardly 
and different and distinctive head standard picture handling libraries and bundles are available
in python which can import and get the yield effectively, for example, Scikit-image, SVD, NumPy
and matplotlib. In addition to that, it is utilized English accentuations which are direct and learn.

# 1.1 Objectives

• The main objective of this practical was to gain brief knowledge about one of the 
applications of the image processing technique of data mining in images in computers using 
Singular Value Decomposition methods. 

• The image compression algorithms were implemented by using the Singular Value 
Decomposition methods and high-level programming language of Python 03.

• Secondly this practical was gain brief knowledge about the root mean square error of the 
reconstructed images using SVD with selected truncation value.
• Furthermore, it was more acquainted with how to use SVD, NumPy, Matplotlib libraries and 
python coding.

# 2 METHODOLOGY

Firstly, the theoretical background of the Singular Value Decomposition was identified, and 
mathematical steps were implemented for image compression before coding. Therefore, the SVD 
of a matrix can be written as
𝐴 = 𝑈∑𝑉^𝑇

where A is input matrix, which was created using image row data of the database, ∑ is singular 
value vector, U and 𝑉^𝑇
is a matrix with orthonormal columns. The truncation value was used to 
compress the image. The below figure is shown the process of taking compressed images[s14320_SVD (Singular Value Decomposition).pdf](https://github.com/Umiisuki/SVD/files/10944399/s14320_SVD.Singular.Value.Decomposition.pdf)
