import sys

import glm
import random
import numpy as np
import cv2 as cv

block_size = 1.0

def getConfig(cam_string):
    cam_xml = cv.FileStorage(f'data\{cam_string}\config.xml', cv.FileStorage_READ)
    cam = cam_xml.getNode('CameraMatrix').mat()
    dist = cam_xml.getNode('DistortionCoeffs').mat()
    rvecs = cam_xml.getNode('rvec').mat()
    tvecs = cam_xml.getNode('tvecs').mat()
    R = cam_xml.getNode('RotationMatrix').mat()
    T = cam_xml.getNode('TranslationMatrix').mat()/100

    #flip the y and z axis and negate z
    T = [T[0], -T[2], T[1]]
    #R = [R[0], -R[2], R[1]]
    print(cam_string)
    print(T)

    return cam, dist, rvecs, tvecs, R, T

#get config for each camera
mtx1, dist1, rvecs1, tvecs1, R1, T1 = getConfig('cam1')
mtx2, dist2, rvecs2, tvecs2, R2, T2 = getConfig('cam2')
mtx3, dist3, rvecs3, tvecs3, R3, T3 = getConfig('cam3')
mtx4, dist4, rvecs4, tvecs4, R4, T4 = getConfig('cam4')

mask1 = cv.imread('data\cam1\mask.png')
mask2 = cv.imread('data\cam2\mask.png')
mask3 = cv.imread('data\cam3\mask.png')
mask4 = cv.imread('data\cam4\mask.png')
video1 = cv.VideoCapture('data/cam1/background.avi')
frame1 = video1.read()
video2 = cv.VideoCapture('data/cam2/background.avi')
frame2 = video2.read()
video3 = cv.VideoCapture('data/cam3/background.avi')
frame3 = video3.read()
video4 = cv.VideoCapture('data/cam4/background.avi')
frame4 = video4.read()

voxelFrameIdx = 0



def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data

def inMask(x, y, z, R, T, mtx, dist, mask, img):
    # project x,y,z to each camera
    # add it to data if it is in every mask
    distance = np.linalg.norm(np.array([x, y, z]) - T)
    point = cv.projectPoints(np.array([[x, y, z]], dtype=np.float32), R, T, mtx, dist)[0][0][0]
    xpoint = point[0]
    ypoint = point[1]
    #check if point is out of bounds
    if xpoint < 0 or xpoint >= mask.shape[0] or ypoint < 0 or ypoint >= mask.shape[1]:
        return False
    # return if point in mask is white
    inMask = (mask[int(xpoint), int(ypoint)] == [255, 255, 255]).all()
    color = img[int(xpoint), int(ypoint)]
    return inMask, distance, color

def generate_voxel_lookup_table(width, height, depth):

    distance = sys.maxint()
    dict = dict()
    for frameIdx in range(video1.get(cv.CAP_PROP_FRAME_COUNT)):
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    print(x,y,z)
                    (incam1, distance1, color1) = inMask(x, y, z, R1, T1, mtx1, dist1, mask1, frame1)
                    if not incam1:
                        dict = {(frameIdx, x, y, z): (False, np.array([0,0,0]))}
                        continue
                    (incam2,distance2, color2) = inMask(x, y, z, R2, T2, mtx2, dist2, mask2, frame2)
                    if not incam2:
                        dict = {(frameIdx, x, y, z): (False, np.array([0,0,0]))}
                        continue
                    (incam3,distance3, color3) = inMask(x, y, z, R3, T3, mtx3, dist3, mask3, frame3)
                    if not incam3:
                        dict = {(frameIdx, x, y, z): (False, np.array([0,0,0]))}
                        continue
                    (incam4, distance4, color4) = inMask(x, y, z, R4, T4, mtx4, dist4, mask4, frame4)
                    if not incam4:
                        dict = {(frameIdx, x, y, z): (False, np.array([0,0,0]))}
                        continue
                    if distance1 < distance:
                        distance = distance1
                        color = color1
                    if distance2 < distance:
                        distance = distance2
                        color = color2
                    if distance3 < distance:
                        distance = distance3
                        color = color3
                    if distance4 < distance:
                        distance = distance4
                        color = color4
                    dict = {(frameIdx, x, y, z): True}
        print(f'frame {frameIdx} done')
    #save dict to file
    np.save('voxel_lookup_table.npy', dict)

def set_voxel_positions(width, height, depth):
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    #get mask.png from every camera
    lookupTable = np.load('voxel_lookup_table.npy').item()
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if lookupTable[(voxelFrameIdx, x, y, z)][0]:
                    color = lookupTable[(voxelFrameIdx, x, y, z)][1]
                    data.append([x * block_size - width / 2, y * block_size - height / 2, z * block_size - depth / 2])
    return data



def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    # return [[-64 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, 63 * block_size],
    #         [63 * block_size, 64 * block_size, -64 * block_size],
    #         [-64 * block_size, 64 * block_size, -64 * block_size]]
    # swap y and z

    return [T1, T2, T3, T4]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [R1, R2, R3, R4]
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # print(cam_rotations[0])
    for c in range(len(cam_rotations)):
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0][0], [1,0,0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1][0], [0,1,0])
        cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2][0], [0,0,1])
    print(cam_rotations[0])
    for c in range(len(cam_rotations)):
        for i in range(3):
            for j in range(3):
                cam_rotations[c][i][j] = cam_angles[c][i][j]
    print(cam_rotations[0])
    # # p
    # print(R1)
    # print(cam_rotations[0])

    return cam_rotations
