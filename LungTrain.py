import argparse
import Models , LoadBatches



parser = argparse.ArgumentParser()
# parser.add_argument("--save_weights_path", type = str,default="weights/ex1"  )
# parser.add_argument("--train_images", type = str ,default="data/dataset1/images_prepped_train/" )
# parser.add_argument("--train_annotations", type = str,default="data/dataset1/annotations_prepped_train/"  )
# parser.add_argument("--n_classes", type=int,default=10 )
# parser.add_argument("--input_height", type=int , default = 320)
# parser.add_argument("--input_width", type=int , default = 640)
# #
# parser.add_argument('--validate',action='store_false')
# parser.add_argument("--val_images", type = str , default = "data/dataset1/images_prepped_test/")
# parser.add_argument("--val_annotations", type = str , default = "data/dataset1/annotations_prepped_test/")
# #
# parser.add_argument("--epochs", type = int, default = 5 )
# parser.add_argument("--batch_size", type = int, default = 2 )
# parser.add_argument("--val_batch_size", type = int, default = 2 )
# parser.add_argument("--load_weights", type = str,default="data/vgg16_weights_th_dim_ordering_th_kernels.h5"  )
# #
# parser.add_argument("--model_name", type = str , default = "vgg_unet")
# parser.add_argument("--optimizer_name", type = str , default = "adadelta")
parser.add_argument("--save_weights_path", type = str,default="weights/ex5"  )
parser.add_argument("--train_images", type = str ,default="/home/ahp/PycharmProjects/tensorflow/tensorflow_tutorials-master/python/Fully_Connected_CRFs/dataset/withoutDisease/newDataset/trainData/" )
parser.add_argument("--train_annotations", type = str,default="/home/ahp/PycharmProjects/tensorflow/tensorflow_tutorials-master/python/Fully_Connected_CRFs/dataset/withoutDisease/newDataset/annTrain/"  )
parser.add_argument("--n_classes", type=int,default=2 )
parser.add_argument("--input_height", type=int , default = 794)
parser.add_argument("--input_width", type=int , default = 802)
#
parser.add_argument('--validate',action='store_false')
parser.add_argument("--val_images", type = str , default = "/home/ahp/PycharmProjects/tensorflow/tensorflow_tutorials-master/python/Fully_Connected_CRFs/dataset/withoutDisease/newDataset/testData/")
parser.add_argument("--val_annotations", type = str , default = "/home/ahp/PycharmProjects/tensorflow/tensorflow_tutorials-master/python/Fully_Connected_CRFs/dataset/withoutDisease/newDataset/annTest/")
#
parser.add_argument("--epochs", type = int, default = 10 )
parser.add_argument("--batch_size", type = int, default = 2 )
parser.add_argument("--val_batch_size", type = int, default = 2 )
parser.add_argument("--load_weights", type = str,default="data/vgg16_weights_th_dim_ordering_th_kernels.h5"  )
#
parser.add_argument("--model_name", type = str , default = "vgg_segnet")
parser.add_argument("--optimizer_name", type = str , default = "adadelta")
# parser.add_argument("--save_weights_path", type = str )
# parser.add_argument("--train_images", type = str  )
# parser.add_argument("--train_annotations", type = str )
# parser.add_argument("--n_classes", type=int,default=10 )
# parser.add_argument("--input_height", type=int , default = 224  )
# parser.add_argument("--input_width", type=int , default = 224 )
#
# parser.add_argument('--validate',action='store_false')
# parser.add_argument("--val_images", type = str )
# parser.add_argument("--val_annotations", type = str )
#
# parser.add_argument("--epochs", type = int, default = 5 )
# parser.add_argument("--batch_size", type = int, default = 2 )
# parser.add_argument("--val_batch_size", type = int, default = 2 )
# parser.add_argument("--load_weights", type = str)
#
# parser.add_argument("--model_name", type = str , default = "vgg_segnet")
# parser.add_argument("--optimizer_name", type = str , default = "adadelta")


args = parser.parse_args()

train_images_path = args.train_images
train_segs_path = args.train_annotations
train_batch_size = args.batch_size
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
validate = args.validate
save_weights_path = args.save_weights_path
epochs = args.epochs
load_weights = args.load_weights

optimizer_name = args.optimizer_name
model_name = args.model_name

if validate:
	val_images_path = args.val_images
	val_segs_path = args.val_annotations
	val_batch_size = args.val_batch_size

modelFns = { 'vgg_segnet':Models.VGGSegnet.VGGSegnet , 'vgg_unet':Models.VGGUnet.VGGUnet , 'vgg_unet2':Models.VGGUnet.VGGUnet2 , 'fcn8':Models.FCN8.FCN8 , 'fcn32':Models.FCN32.FCN32   }
modelFN = modelFns[ model_name ]

m = modelFN( n_classes , input_height=input_height, input_width=input_width   )
m.compile(loss='categorical_crossentropy',
      optimizer= optimizer_name ,
      metrics=['accuracy'])


#if len( load_weights ) > 0:
#	m.load_weights(load_weights)


print "Model output shape" ,  m.output_shape

output_height = m.outputHeight
output_width = m.outputWidth

G  = LoadBatches.imageSegmentationGenerator( train_images_path , train_segs_path ,  train_batch_size,  n_classes , input_height , input_width , output_height , output_width   )


if validate:
	G2  = LoadBatches.imageSegmentationGenerator( val_images_path , val_segs_path ,  val_batch_size,  n_classes , input_height , input_width , output_height , output_width   )

if not validate:
	for ep in range( epochs ):
		m.fit_generator( G , 5  , epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep ) )
		m.save( save_weights_path + ".model." + str( ep ) )
else:
	for ep in range( epochs ):
		m.fit_generator( G , 100  , validation_data=G2 , validation_steps=5 ,  epochs=1 )
		m.save_weights( save_weights_path + "." + str( ep )  )
		m.save( save_weights_path + ".model." + str( ep ) )


