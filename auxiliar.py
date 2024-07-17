import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
from ultralytics import YOLO
import numpy as np
import logging


logging.basicConfig(filename= 'logs/log.log',
                    filemode= 'w',
                    level= 'INFO')

def process_image(image_path, model_path, save_dir, project_name):
    model = YOLO(model_path)
    result = model(source=image_path, conf=0.5, show=False, save=False, project=save_dir, name=project_name)
    
    # extract keypoints
    keypoints = result[0].keypoints.xy[0].numpy()
    
    image_bef = Image.open(image_path)
    image_aft_path = f'{save_dir}/{project_name}/referee.png'
    image_aft = Image.open(image_aft_path)
    
    plt.figure(figsize=(22, 12))
    # original
    plt.subplot(1, 3, 1)
    plt.imshow(image_bef)
    plt.title('Original Image')
    # keypoints image
    plt.subplot(1, 3, 2)
    plt.imshow(image_bef)
    for i, point in enumerate(keypoints):
        x, y = point
        plt.gca().add_patch(patches.Circle((x, y), radius=2, color='red'))
    plt.title('YOLO Keypoints')
    # processed image
    plt.subplot(1, 3, 3)
    plt.imshow(image_aft)
    plt.title('YOLO Image')

    save_path = f'{save_dir}/{project_name}/referee_processed.png'
    plt.savefig(save_path)
    # print(f'Image saved in: {save_path}')


    keypoints = result[0].keypoints.xy[0].numpy()
    indexed_xy = np.array([(f'{i+1}: {x}', y) for i, (x, y) in enumerate(keypoints)])
    logging.info(f'Keypoints coordinates are: {indexed_xy}')

    plt.figure(figsize=(12,6))
    plt.imshow(image_bef)
    for i, point in enumerate(keypoints):
        x, y = point
        plt.gca().add_patch(patches.Circle((x, y), radius=5, color='red'))
        plt.text(x, y, str(i+1), fontsize=8, color='black', ha='center', va='center')

    save_path_keypoints = f'{save_dir}/{project_name}/referee_keypoints.png'
    plt.savefig(save_path_keypoints)
    # print(f'Image saved in: {save_path_keypoints}')

    # 'distangle' between two sample keypoints
    A = keypoints[5]
    B = keypoints[9]
    distance = np.linalg.norm(A - B)

    dy = B[1] - A[1]
    dx = B[0] - A[0]
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    logging.info(f"Distance between keypoints: {distance}")
    logging.info(f"Angle between keypoints (degrees): {angle_deg}")

    # Points A - B
    plt.plot(A[0], A[1], 'ro')
    plt.text(A[0],A[1]-10,'A',color='red')
    plt.plot(B[0], B[1], 'ro')
    plt.text(B[0],B[1]-10,'B',color='red')

    # Distance between points
    plt.plot([A[0], B[0]], [A[1], B[1]], 'g--', label='Distance between points')
    # Angle between points
    angle_text = f'Angle: \n{angle_deg:.2f} degrees'
    plt.text((A[0] + B[0])/2,75, angle_text, ha='center', va='baseline', color='purple')
    plt.legend()
    save_path_distangle = f'{save_dir}/{project_name}/referee_keypoints_distangle.png'
    plt.savefig(save_path_distangle)


    # # Definir los límites de la región ampliada (ZOOM)
    # x_margin = 50
    # y_margin = 50
    # x_min = min(A[0], B[0]) - x_margin
    # x_max = max(A[0], B[0]) + x_margin
    # y_min = min(A[1], B[1]) - y_margin
    # y_max = max(A[1], B[1]) + y_margin

    # # Establecer los límites de los ejes x e y
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_max, y_min)  # El límite inferior es mayor que el superior para invertir el eje y (imagen invertida)

    # # Mostrar la región ampliada de la imagen
    # plt.imshow(image_bef)

    # # Points A - B
    # plt.plot(A[0], A[1], 'ro')
    # plt.text(A[0], A[1] - 10, 'A', color='red')
    # plt.plot(B[0], B[1], 'ro')
    # plt.text(B[0], B[1] - 10, 'B', color='red')

    # # Distance between points
    # plt.plot([A[0], B[0]], [A[1], B[1]], 'g--', label='Distance between points')

    # # Angle between points
    # angle_text = f'Angle: \n{angle_deg:.2f} degrees'
    # plt.text((A[0] + B[0]) / 2, 75, angle_text, ha='center', va='baseline', color='purple')

    # plt.legend()
    # plt.show()


