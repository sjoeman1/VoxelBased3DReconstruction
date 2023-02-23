import glm
import random
import numpy as np
import cv2 as cv

block_size = 1.0

def getConfig(cam_string):
    cam_xml = cv.FileStorage(f'data\{cam_string}\config.xml', cv.FileStorage_READ)
    cam = cam_xml.getNode('CameraMatrix').mat()
    dist = cam_xml.getNode('DistortionCoeffs').mat()
    rot = cam_xml.getNode('RotationMatrix').mat()
    trans = cam_xml.getNode('TranslationMatrix').mat()
    print(trans)

    return cam, dist, rot, trans

#get config for each camera
mtx1, dist1, rvecs1, tvecs1 = getConfig('cam1')
mtx2, dist2, rvecs2, tvecs2 = getConfig('cam2')
mtx3, dist3, rvecs3, tvecs3 = getConfig('cam3')
mtx4, dist4, rvecs4, tvecs4 = getConfig('cam4')



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
    #get mask.png from every camera
    mask1 = cv.imread('data\cam1\mask.png')
    mask2 = cv.imread('data\cam2\mask.png')
    mask3 = cv.imread('data\cam3\mask.png')
    mask4 = cv.imread('data\cam4\mask.png')
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                print(x,y,z)
                incam1 = inMask(x, y, z, rvecs1, tvecs1, mtx1, dist1, mask1)
                if not incam1:
                    continue
                incam2 = inMask(x, y, z, rvecs2, tvecs2, mtx2, dist2, mask2)
                if not incam2:
                    continue
                incam3 = inMask(x, y, z, rvecs3, tvecs3, mtx3, dist3, mask3)
                if not incam3:
                    continue
                incam4 = inMask(x, y, z, rvecs4, tvecs4, mtx4, dist4, mask4)
                if not incam4:
                    continue
                data.append([x * block_size - width / 2, y * block_size - height / 2, z * block_size - depth / 2])

    return data


def inMask(x, y, z, rvecs, tvecs, mtx, dist, mask):
    # project x,y,z to each camera
    # add it to data if it is in every mask
    point = cv.projectPoints(np.array([[x, y, z]], dtype=np.float32), rvecs, tvecs, mtx, dist)[0][0][0]
    xpoint = point[0]
    ypoint = point[1]
    #check if point is out of bounds
    if xpoint < 0 or xpoint >= mask.shape[0] or ypoint < 0 or ypoint >= mask.shape[1]:
        return False
    # return if point in mask is white
    return (mask[int(xpoint), int(ypoint)] == [255, 255, 255]).all()


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]]
    return [tvecs1, tvecs2, tvecs3, tvecs4]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [rvecs1, rvecs2, rvecs3, rvecs4]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0], [1, 0, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1], [0, 1, 0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2], [0, 0, 1])

    return cam_rotations
