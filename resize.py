import PIL
from PIL import Image
#basewidth = 300
import glob
import pickle
import cv2
import numpy as np

'''
for i in glob.glob("../traditional-decor-patterns/decor/*.png"):
	img=Image.open(i)
	img=img.resize((112,112), PIL.Image.ANTIALIAS)
	img.save("resized_images/resized_image"+i.split("/")[-1])


'''


resized_images = []
for i in glob.glob("resized_images/*.png"):
	resized_images.append(cv2.imread(i))

resized_images = np.array(resized_images)
pickle.dump(resized_images,open("resized_images.pkl","wb"))

'''
a = pickle.load(open("resized_images.pkl","rb"))
print(np.array(a).shape)
'''