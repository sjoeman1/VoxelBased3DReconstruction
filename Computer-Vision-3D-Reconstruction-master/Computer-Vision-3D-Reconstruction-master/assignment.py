import sys

import glm
import random
import numpy as np
import cv2 as cv

block_size = 1.0

voxel_scale = 23


def getConfig(cam_string):
    cam_xml = cv.FileStorage(f'data\{cam_string}\config.xml', cv.FileStorage_READ)
    cam = cam_xml.getNode('CameraMatrix').mat()
    dist = cam_xml.getNode('DistortionCoeffs').mat()
    rvecs = cam_xml.getNode('rvec').mat()
    tvecs = cam_xml.getNode('tvecs').mat()
    R = cam_xml.getNode('RotationMatrix').mat()
    T = (cam_xml.getNode('TranslationMatrix').mat() / voxel_scale)


    return cam, dist, rvecs, tvecs, R, T


# get config for each camera
mtx1, dist1, rvecs1, tvecs1, R1, T1 = getConfig('cam1')
mtx2, dist2, rvecs2, tvecs2, R2, T2 = getConfig('cam2')
mtx3, dist3, rvecs3, tvecs3, R3, T3 = getConfig('cam3')
mtx4, dist4, rvecs4, tvecs4, R4, T4 = getConfig('cam4')

def load_avi(cam_number, name):
    # load frames from avi file
    frames = []
    cap = cv.VideoCapture('data/cam' + str(cam_number) + '/' + name + '.avi')
    return cap

masks1 = load_avi(1, 'masks')
masks2 = load_avi(2, 'masks')
masks3 = load_avi(3, 'masks')
masks4 = load_avi(4, 'masks')
mask_height, mask_width = int(masks1.get(cv.CAP_PROP_FRAME_HEIGHT)), int(masks1.get(cv.CAP_PROP_FRAME_WIDTH))


frames1 = load_avi(1, 'frames')
frames2 = load_avi(2, 'frames')
frames3 = load_avi(3, 'frames')
frames4 = load_avi(4, 'frames')



voxelFrameIdx = 0

clicks = 0

lookupTable = {}



def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors

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



    data = []
    #create numpy array for colors
    colors = np.zeros((width, height, depth, 3))
    print("frame: " + str(clicks))
    # get mask from every camera
    masks = [masks1.read()[1], masks2.read()[1], masks3.read()[1], masks4.read()[1]]
    frames = [frames1.read()[1], frames2.read()[1], frames3.read()[1], frames4.read()[1]]
    clicks += 1

    # create a zerros array of the shape width, height, depth
    countInShapes = np.zeros((width, height, depth))

    for c in range(4):
        for x in range(mask_width):
            for y in range(mask_height):
                if (masks[c][y][x] != 255).all():
                    continue
                if (c, x, y) in lookupTable:
                    voxels = lookupTable[(c, x, y)]
                    for v in voxels:
                        vx, vy, vz = v
                        # add one to countInShapes at the voxel position
                        colors[int(vx)][int(vy)][int(vz)] = frames[c][y][x]
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
    data2 = data.copy()
    print("coloring model")
    colors = create_colors(data2, width, height, depth, colors)
    print("done")
    return data, colors

def create_colors(data, width, height, depth, colors):
    countOccluded = np.zeros((width, height, depth))
    data = np.array(data)
    #check for each voxel in the data if it is the closest voxel to the camera
    for c in range(4):
        color = np.zeros((width, height, depth, 3))
        for x,y,z in data:
            #check if neighbor in direction of camera is occluded

            #get vector from voxel to camera
            cam_pos, _ = get_cam_positions()
            cam_pos = np.array([cam_pos[c][0][0], cam_pos[c][1][0], cam_pos[c][2][0]])
            voxel_pos = np.array([x, y, z])
            vec = cam_pos - voxel_pos

            #normalize vector
            vec = vec / np.linalg.norm(vec)

            #check if neighbor in direction of vector is occluded
            neighbor = np.zeros(3)
            neighbor = voxel_pos + vec
            #round vector to ints
            neighbor_index = neighbor.astype(int)
            limit = 3
            i = 0
            #check if neighbor is in data
            found = False
            while not found:
                if np.any(np.all(data == neighbor_index, axis=1)):
                    found = True
                if found:
                    break
                neighbor += vec
                neighbor_index = neighbor.astype(int)
                i += 1
                if i > limit:
                    # if no neighbor is found, we give a color
                    color[x][y][z] = colors[x][y][z] /255
                    break
            if found:
                if countOccluded[neighbor_index[0]][neighbor_index[1]][neighbor_index[2]] == 0:
                   color[x][y][z] = [0,0,255]
                   countOccluded[x][y][z] += 1

    #filter countOccluded to only exclude voxels that are occluded by all cameras
    occluded = np.where(countOccluded > 3, 1, 0)
    #change color of voxels that are occluded by all cameras to black
    result = []
    for x,y,z in data:
                if occluded[x][y][z]:
                    result.append([0,0,0])
                else:
                    result.append(color[x][y][z])
    return result


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.
    return [T1, T2, T3, T4], [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    cam_angles = [R1, R2, R3, R4]
    cam_rotations = []
    for c in range(len(cam_angles)):
        #rotate 90 degrees around y axis
        cam_rotations.append(glm.rotate(glm.mat4(glm.mat3(cam_angles[c])), glm.radians(-90), glm.vec3(0, 1, 0)))

    return cam_rotations
