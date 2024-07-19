from ultralytics import YOLO
import auxiliar
import yaml
import os


# load config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# load pose model
model_name = config['pose_model']['name']
model_folder = config['pose_model']['folder']
model_path = os.path.join(model_folder, model_name)
model = YOLO(model_path)


# First, we try with a single image to extrect keypoints and pose.
image_path = config['image']
auxiliar.process_image(image_path, model_path, save_dir='results', project_name='referee_pose')



# # video = config['videos']['bar_backflip']
# # result = model(source = video, conf=0.5, show=True, save=True, project="results", name='bar_backflip')
