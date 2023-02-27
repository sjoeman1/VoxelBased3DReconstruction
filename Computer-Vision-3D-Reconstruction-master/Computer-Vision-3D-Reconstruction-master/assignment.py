import sys

import glm
import random
import numpy as np
import cv2 as cv

block_size = 1.0

scaling = 50

def getConfig(cam_string):
    cam_xml = cv.FileStorage(f'data\{cam_string}\config.xml', cv.FileStorage_READ)
    cam = cam_xml.getNode('CameraMatrix').mat()
    dist = cam_xml.getNode('DistortionCoeffs').mat()
    rvecs = cam_xml.getNode('rvec').mat()
    tvecs = cam_xml.getNode('tvecs').mat()
    R = cam_xml.getNode('RotationMatrix').mat()
    T = (cam_xml.getNode('TranslationMatrix').mat() / scaling)

    #flip the y and z axis and negate z
    T = [T[0], -T[2], T[1]]
    # rotate R by 180 degrees around y axis
    R = R * np.matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    #rotete R by -90 degrees around z axis
    R = R * np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])


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

lookupTable = {}



def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x*block_size - width/2, -block_size, z*block_size - depth/2])
    return data



def generate_voxel_lookup_table(width, height, depth):
    # generate meshgrid of width, height, depth and project to each camera
    global lookupTable
    objpts = np.float32(np.mgrid[0:width, 0:height, 0:depth].T.reshape(-1, 3))
    print(objpts)
    print(rvecs1)
    # project to each camera
    imgpts1 = cv.projectPoints(objpts, R1, tvecs1, mtx1, dist1)[0]
    #reshape to 3D array
    imgpts1 = imgpts1.reshape(width, height, depth, 2)
    imgpts2 = cv.projectPoints(objpts, R2, tvecs2, mtx2, dist2)[0]
    imgpts2 = imgpts2.reshape(width, height, depth, 2)
    imgpts3 = cv.projectPoints(objpts, R3, tvecs3, mtx3, dist3)[0]
    imgpts3 = imgpts3.reshape(width, height, depth, 2)
    imgpts4 = cv.projectPoints(objpts, R4, tvecs4, mtx4, dist4)[0]
    imgpts4 = imgpts4.reshape(width, height, depth, 2)
    print(imgpts1)
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                lookupTable[x,y,z] = [imgpts1[x,y,z], imgpts2[x,y,z], imgpts3[x,y,z], imgpts4[x,y,z]]
        print(x)





    #save dict to file
    print("saving lookup table")
    np.save('voxel_lookup_table.npy', lookupTable)

def set_voxel_positions(width, height, depth):
    global lookupTable
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    #get mask.png from every camera
    print("generating")
    if not lookupTable:
        print("loading lookup table")
        generate_voxel_lookup_table(width, height, depth)
    cv.imshow('mask', mask2)
    data = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                imgpts = lookupTable[x, y, z]
                inAll, color = inMask(imgpts, [mask1, mask2, mask3, mask4], [frame1, frame2, frame3, frame4])
                if inAll:
                    data.append([x*block_size - width/2, y*block_size, z*block_size - depth/2])

    return data

def inMask(imgpt, masks, imgs):
    # project x,y,z to each camera
    # add it to data if it is in every mask
    # print(masks[0][int(295)][int(351)])
    color = np.zeros(3)
    for i in range(1):
        ypoint = imgpt[i][0]
        xpoint = imgpt[i][1]
        #check if point is out of bounds
        if xpoint < 0 or xpoint >= masks[0].shape[1] or ypoint < 0 or ypoint >= masks[0].shape[0]:
            return False, color
        # return if point in mask is white
        inMask = (masks[i][int(xpoint)][int(ypoint)] == 255).all()
        if not inMask:
            return False, color

        #take 25 procent of the img color
        # color += imgs[i][int(xpoint)][int(ypoint)]
    # return if point in mask is white

    return True, color



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
    # # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    # cam_angles = [rvecs1, rvecs2, rvecs3, rvecs4]
    # cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0], [1, 0, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1], [0, 1, 0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2], [0, 0, 1])
    #
    # return cam_rotations
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [R1, R2, R3, R4]
    cam_rotations = []
    # # print(cam_rotations[0])
    # for c in range(len(cam_rotations)):
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][0][0], [1,0,0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][1][0], [0,1,0])
    #     cam_rotations[c] = glm.rotate(cam_rotations[c], cam_angles[c][2][0], [0,0,1])
    # print(cam_rotations[0])
    for c in range(len(cam_angles)):
        cam_rotations.append(glm.mat4(glm.mat3(cam_angles[c])))
    print(cam_rotations[0])
    # # p
    # print(R1)
    # print(cam_rotations[0])

    return cam_rotations
