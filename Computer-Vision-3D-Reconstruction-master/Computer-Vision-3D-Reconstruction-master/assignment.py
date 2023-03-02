import sys

import glm
import random
import numpy as np
import cv2 as cv

block_size = 1.0

scaling = 50

voxel_scale = 27


def getConfig(cam_string):
    cam_xml = cv.FileStorage(f'data\{cam_string}\config.xml', cv.FileStorage_READ)
    cam = cam_xml.getNode('CameraMatrix').mat()
    dist = cam_xml.getNode('DistortionCoeffs').mat()
    rvecs = cam_xml.getNode('rvec').mat()
    tvecs = cam_xml.getNode('tvecs').mat()
    R = cam_xml.getNode('RotationMatrix').mat()
    T = (cam_xml.getNode('TranslationMatrix').mat() / scaling)

    return cam, dist, rvecs, tvecs, R, T


# get config for each camera
mtx1, dist1, rvecs1, tvecs1, R1, T1 = getConfig('cam1')
mtx2, dist2, rvecs2, tvecs2, R2, T2 = getConfig('cam2')
mtx3, dist3, rvecs3, tvecs3, R3, T3 = getConfig('cam3')
mtx4, dist4, rvecs4, tvecs4, R4, T4 = getConfig('cam4')

def loadMasks(cam_string):
    masks = []
    #load masks from avi file
    cap = cv.VideoCapture(f'data\{cam_string}\masks.avi')
    return cap

masks1 = loadMasks('cam1')
mask_height, mask_width = int(masks1.get(cv.CAP_PROP_FRAME_HEIGHT)), int(masks1.get(cv.CAP_PROP_FRAME_WIDTH))
masks2 = loadMasks('cam2')
masks3 = loadMasks('cam3')
masks4 = loadMasks('cam4')

frames1 = load_avi(1, 'frames')
frames2 = load_avi(2, 'frames')
frames3 = load_avi(3, 'frames')
frames4 = load_avi(4, 'frames')

video1 = cv.VideoCapture('data/cam1/background.avi')
video2 = cv.VideoCapture('data/cam2/background.avi')
video3 = cv.VideoCapture('data/cam3/background.avi')
video4 = cv.VideoCapture('data/cam4/background.avi')
videos = [video1, video2, video3, video4]

voxelFrameIdx = 0

lookupTable = {}



def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data = []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
    return data


def generate_voxel_lookup_table(width, height, depth):
    # generate meshgrid of width, height, depth and project to each camera
    global lookupTable

    print("generating lookup table")
    objpts = np.float32(np.mgrid[-width / 2:width / 2, 0:height, -depth / 2:depth / 2].T.reshape(-1, 3))

    objpts = objpts * voxel_scale
    rvec1 = cv.Rodrigues(R1)[0]
    rvec2 = cv.Rodrigues(R2)[0]
    rvec3 = cv.Rodrigues(R3)[0]
    rvec4 = cv.Rodrigues(R4)[0]
    # project to each camera
    imgpts1 = cv.projectPoints(objpts, rvec1, tvecs1, mtx1, dist1)[0]
    imgpts2 = cv.projectPoints(objpts, rvec2, tvecs2, mtx2, dist2)[0]
    imgpts3 = cv.projectPoints(objpts, rvec3, tvecs3, mtx3, dist3)[0]
    imgpts4 = cv.projectPoints(objpts, rvec4, tvecs4, mtx4, dist4)[0]
    imgpts = [imgpts1, imgpts2, imgpts3, imgpts4]
    # save to dictg

    for c in range(4):
        # iterate over imgpts and append objpts to lookupTable[c,x,y]
        for i in range(len(imgpts[c])):
            x = int(imgpts[c][i][0][0])
            y = int(imgpts[c][i][0][1])
            if x < 0 or x >= mask_width or y < 0 or y >= mask_height:
                continue
            if lookupTable.get((c, x, y)) is None:
                lookupTable[(c, x, y)] = []
            lookupTable[(c, x, y)].append([objpts[i][0]*block_size / voxel_scale,
                                           objpts[i][1]*block_size / voxel_scale,
                                           objpts[i][2]*block_size / voxel_scale])
    #save dict to file
    print("saving lookup table")
    np.save('voxel_lookup_table.npy', lookupTable)
    print("done")


def set_voxel_positions(width, height, depth):
    global lookupTable
    global clicks
    global voxel_scale

    if not lookupTable:
        lookupTable = np.load('voxel_lookup_table.npy', allow_pickle=True).item()
        print("loaded lookup table")
    # Generates random voxel locations
    # TODO: You need to calculate proper voxel arrays instead of random ones.
    # get mask.png from every camera
    data = []
    print("frame: " + str(clicks))
    masks = [masks1.read()[1], masks2.read()[1], masks3.read()[1], masks4.read()[1]]
    frames = [video1.read()[1], video2.read()[1], video3.read()[1], video4.read()[1]]
    clicks += 1

    # create an zerros array of the shape width, height, depth
    countInShapes = np.zeros((width, height, depth))

    for c in range(4):
        for x in range(mask_width):
            for y in range(mask_height):
                if (masks[c][y][x] != 255).all():
                    continue
                if (c, x, y) in lookupTable:
                    voxels = lookupTable[(c, x, y)]
                    color = get_color(c, y, x, frames)
                    for v in voxels:
                        vx, vy, vz = v
                        # add one to countInShapes at the voxel position
                        countInShapes[int(vx)][int(vy)][int(vz)] += 1
        print(f'finished camera {c}')

    # filter count in shapes to only include voxels that are in all masks
    # inAllCams = (countInShapes == 4).nonzero()
    # data = np.array(inAllCams[0], inAllCams[1], inAllCams[2]).T
    inAllCams = np.where(countInShapes == 4, 1, 0)
    # add voxels in countInShapes to data
    for x in range(int(-width/2), int(width/2)):
        for y in range(height):
            for z in range(int(-depth/2), int(depth/2)):
                if inAllCams[x][y][z] == 1:
                    data.append([x, y, z])


    return data
    # return data, colors


def get_color(c, y, x, frames):
    color = np.zeros(3)

    color = frames[c][y][x]

    return color


def inMask(imgpt, masks, imgs):
    # project x,y,z to each camera
    # add it to data if it is in every mask
    # print(masks[0][int(295)][int(351)])
    color = np.zeros(3)
    for i in range(3, 4):
        xpoint = int(imgpt[i][0])
        ypoint = int(imgpt[i][1])
        # check if point is out of bounds
        if xpoint < 0 or xpoint >= masks[0].shape[1] or ypoint < 0 or ypoint >= masks[0].shape[0]:
            return False, color
        # return if point in mask is white
        inMask = (masks[i][ypoint][xpoint] == 255).all()
        if not inMask:
            return False, color

        # take 25 procent of the img color
        # color += imgs[i][int(xpoint)][int(ypoint)]
    # return if point in mask is white

    return True, color


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [T1, T2, T3, T4]


def get_cam_rotation_matrices():
    # # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [R1, R2, R3, R4]
    cam_rotations = []
    for c in range(len(cam_angles)):
        cam_rotations.append(glm.mat4(glm.mat3(cam_angles[c])))

    return cam_rotations
