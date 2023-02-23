import glm
import random
import numpy as np
import cv2 as cv

block_size = 1.0

cam1_xml = cv.FileStorage('data\cam1\config.xml', cv.FileStorage_READ)
cam2_xml = cv.FileStorage('data\cam2\config.xml', cv.FileStorage_READ)
cam3_xml = cv.FileStorage('data\cam3\config.xml', cv.FileStorage_READ)
cam4_xml = cv.FileStorage('data\cam4\config.xml', cv.FileStorage_READ)

def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data


def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if random.randint(0, 1000) < 5:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])
    return data


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    # return the translation matrix for each camera

    return [cam1_xml.getNode('TranslationMatrix'),
            cam2_xml.getNode('TranslationMatrix'),
            cam3_xml.getNode('TranslationMatrix'),
            cam4_xml.getNode('TranslationMatrix')]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # get camera rotation matrices from config.xml
    cam_angles = [cam1_xml.getNode('RotationMatrix'),
                  cam2_xml.getNode('RotationMatrix'),
                  cam3_xml.getNode('RotationMatrix'),
                  cam4_xml.getNode('RotationMatrix')]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0] * np.pi / 180, [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1] * np.pi / 180, [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2] * np.pi / 180, [0, 0, 1])
    return cam_rotations
