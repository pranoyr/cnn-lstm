import os 
import cv2

image_folder='image_data'
video_folder='video_data'
# detete image folder
os.system("rm -rf "+image_folder)
# create image folder
os.system("mkdir "+image_folder)

for label in os.listdir(os.path.join(video_folder)):
	# make label dir
	os.system("mkdir "+os.path.join(image_folder,label))
	for video_name in os.listdir(os.path.join(video_folder,label)):
		i=0
		# make folder
		os.system("mkdir "+os.path.join(image_folder,label,video_name.split('.')[0]))
		camera=cv2.VideoCapture(os.path.join(video_folder,label,video_name))
		# read frames
		while True:
			ret,img=camera.read()
			if ret==False:
				break   
			img_name=str(i)+'.jpg'
			print(os.path.join(image_folder,label,video_name.split('.')[0],img_name))
			# save frames
			cv2.imwrite(os.path.join(image_folder,label,video_name.split('.')[0],img_name),img)
			i+=1
          
          