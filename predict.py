import argparse
import Models , LoadBatches
from keras.models import load_model
import glob
import cv2
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--save_weights_path", type = str,default="weights/ex1")
parser.add_argument("--epoch_number", type = int, default = 0 )
parser.add_argument("--test_images", type = str , default = "data/dataset1/images_prepped_test/")
parser.add_argument("--output_path", type = str , default = "data/predictions/")
parser.add_argument("--input_height", type=int , default = 320  )
parser.add_argument("--input_width", type=int , default = 640 )
parser.add_argument("--model_name", type = str , default = "vgg_unet")
parser.add_argument("--n_classes", type=int,default=10 )

args = parser.parse_args()

n_classes = args.n_classes
model_name = args.model_name
images_path = args.test_images
input_width =  args.input_width
input_height = args.input_height
epoch_number = args.epoch_number

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.load_weights(  args.save_weights_path + "." + str(  epoch_number )  )
m.compile(loss='categorical_crossentropy',
      optimizer= 'adadelta' ,
      metrics=['accuracy'])


output_height = m.outputHeight
output_width = m.outputWidth

images = glob.glob( images_path + "*.jpg"  ) + glob.glob( images_path + "*.png"  ) +  glob.glob( images_path + "*.jpeg"  )
images.sort()

colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

for imgName in images:
	outName = imgName.replace( images_path ,  args.output_path )
	X = LoadBatches.getImageArr(imgName , args.input_width  , args.input_height  )
	pr = m.predict( np.array([X]) )[0]
	pr = pr.reshape(( output_height ,  output_width , n_classes ) ).argmax( axis=2 )
	seg_img = np.zeros( ( output_height , output_width , 3  ) )
	for c in range(n_classes):
		seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
		seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
		seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')
	seg_img = cv2.resize(seg_img  , (480 , 360 ))
	cv2.imwrite(  outName , seg_img )

