import cv2 as cv
import numpy
import numpy as np

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

camIntrinsic1 = cv.VideoCapture('data/cam1/intrinsics.avi')
camIntrinsic2 = cv.VideoCapture('data/cam2/intrinsics.avi')
camIntrinsic3 = cv.VideoCapture('data/cam3/intrinsics.avi')
camIntrinsic4 = cv.VideoCapture('data/cam4/intrinsics.avi')

camExtrinsic1 = cv.VideoCapture('data/cam1/checkerboard.avi')
camExtrinsic2 = cv.VideoCapture('data/cam2/checkerboard.avi')
camExtrinsic3 = cv.VideoCapture('data/cam3/checkerboard.avi')
camExtrinsic4 = cv.VideoCapture('data/cam4/checkerboard.avi')

cam1xml = cv.FileStorage(f'data/cam1/config.xml', cv.FileStorage_READ)
cam2xml = cv.FileStorage(f'data/cam2/config.xml', cv.FileStorage_READ)
cam3xml = cv.FileStorage(f'data/cam3/config.xml', cv.FileStorage_READ)
cam4xml = cv.FileStorage(f'data/cam4/config.xml', cv.FileStorage_READ)

CB_data = cv.FileStorage('data\checkerboard.xml', cv.FileStorage_READ)

s = cv.FileStorage('data\checkerboard.xml', cv.FileStorage_READ)

#chessboard parameters
columns = int(s.getNode('CheckerBoardWidth').real())
rows = int(s.getNode('CheckerBoardHeight').real())
board_shape = (columns, rows)
cube_size = int(s.getNode('CheckerBoardSquareSize').real())
s.release()

# prepare object points with cube size, like (0,0,0), (115,0,0), (230,0,0) ....,(345,575,0)
objp = np.zeros((columns * rows, 3), np.float32)
objp[:, :2] = cube_size * np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

#amount of clicks for the corners
clicks = 0

# #draw a cube on the image given the corners and the projected points
# project 3D points to image plane
def draw_cube(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)


    #first the x and y origin axis lines
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[1]), (255,255,0), 10)
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[3]), (0,255,255), 10)

    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)

    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255,0,0), 6)

    # then draw z origin axis lines
    img = cv.line(img, tuple(imgpts[0]), tuple(imgpts[4]), (255,0,255), 10)

    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)

    return img


# generates a new image from img, with cornerpoints and a cube projected on to it
def  Online(img, calibration, corners = None):

    # the 3d points of a cube. 2 * cube_size to make the cube 2 chessboard squares sized
    length = 4 * cube_size
    cube = np.float32([[0, 0, 0], [0, length, 0], [length, length, 0], [length, 0, 0],
                       [0, 0, -length], [0, length, -length], [length, length, -length], [length, 0, -length]])

    # the camera calibration matrices
    ret, mtx, dist, rvecs, tvecs = calibration

    #gray the image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #find corners
    if corners is None:
        ret, corners = cv.findChessboardCorners(gray, board_shape, flags= cv.CALIB_CB_FAST_CHECK)
        img = cv.drawChessboardCorners(img, board_shape, corners, ret)

    if ret is not False:
        # refine corners
        # corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # find the rotation and translation vectors.
        ret, rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)

        #project the real cube coordinates to image coordinates
        imgpts, jac = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)


        #draw the cube using the coordinates
        img = draw_cube(img, imgpts)



    return img

# calculate the reprojection error by projecting the 3d points to 2d points and measuring the average distance
def reprojectionError(objectpoints, corners, image):
    mean_error = 0
    objp = []
    imgp = []
    objp.append(objectpoints)
    imgp.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objp, imgp, image.shape[::-1], None, None)

    for i in range(len(objp)):
        imgpoints2, _ = cv.projectPoints(objp[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgp[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    return mean_error/len(objp)

# use a projective matrix to calculate all the corners in a grid
# this way both the x and y coordinates are taken into account
def interpolateCorners(init_corners, image):

    #calculate the projective matrix
    input_pts = np.float32(init_corners)
    output_pts = np.float32([[500, 500], [500, 0], [0, 0], [0, 500]])
    M = cv.getPerspectiveTransform(output_pts, input_pts)

    # calculate coordinates of grid between 0 and 500 with length of rows and columns, this number does not matter as it is only the size of the unit matrix
    x = np.linspace(0, 500, rows)
    y = np.linspace(0, 500, columns)
    #combine x and y to get a grid of coordinates
    grid = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)

    corners = []
    #use the matrix to transform the grid to coordinates on the image
    for point in grid:
        corner = cv.perspectiveTransform(np.array([[point]], dtype=np.float32), M)[0]
        corners.append(corner)

    corners = np.array(corners, ndmin=3, dtype=np.float32)
    return corners


# click event handler
# saves the x and y coordinates of 4 clicks
# need to in order: top left, clockwise
def click_event(event, x, y, flags, params):
    global clicks
    corners = params[0]
    if event == cv.EVENT_LBUTTONDOWN:
        print(x, ' ', y)
        corners[clicks] = [x, y]

        clicks += 1
        if clicks == 4:
            clicks = 0
            cv.destroyWindow('Click corners (Clockwise from top-left)')


# shows the image where corners need to be manually acquired
def getChessboardCorners(img):
    cv.imshow('Click corners (Clockwise from top-left)', img)
    corners = np.zeros((4, 2))
    cv.setMouseCallback('Click corners (Clockwise from top-left)', click_event, param= (corners, clicks))
    cv.waitKey(0)
    return True, corners


# collect all the cornerpoints of the images in images
# when corners cant be detected automatically, acquire them manually
# also refines the corner positions of both automatic and manual corners
def Offline(images):

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # get the image and gray them out
    gray = cv.cvtColor(images[0], cv.COLOR_BGR2GRAY)


    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners automatically
        ret, corners = cv.findChessboardCorners(gray, board_shape, flags=cv.CALIB_CB_FAST_CHECK)
        if not ret:
            #find chessboard corners manually
            ret, corners = getChessboardCorners(gray)
            corners = interpolateCorners(corners, img)

        # refine the corner positions
        corners = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        #check quality of image

        error = reprojectionError(objp, corners, gray)
        print(error)
        # If found to be within the error threshold, add object points, image points (after refining them)
        if error < 0.06:
            objpoints.append(objp)
            imgpoints.append(corners)

        # Draw and display the corners
        # cv.drawChessboardCorners(img, board_shape, corners, ret)
        # cv.imshow('img', img)
        # cv.waitKey(0)

    # cv.destroyWindow('img')

    #calibrate the camera using all the points found in all the images
    calibration = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return calibration

def getFrames(cam, frame_count):
    frames = []
    totalFrames = cam.get(cv.CAP_PROP_FRAME_COUNT)
    print(totalFrames)
    _, img = cam.read()
    for i in numpy.linspace(0, totalFrames, num= frame_count, dtype= int, endpoint=False):
        cam.set(cv.CAP_PROP_POS_FRAMES, i)
        ret = False
        while not ret:
            cam_ret, img = cam.read()
            if not cam_ret:
                break
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners automatically
            ret, corners = cv.findChessboardCorners(gray, board_shape, flags=cv.CALIB_CB_FAST_CHECK)
        if ret:
            print(cam.get(cv.CAP_PROP_POS_FRAMES))
            frames.append(img)
    return np.array(frames)



def main():
    img1 = calibrateCam(camIntrinsic1, camExtrinsic1, 'cam1')
    img2 = calibrateCam(camIntrinsic2, camExtrinsic2, 'cam2')
    img3 = calibrateCam(camIntrinsic3, camExtrinsic3, 'cam3')
    img4 = calibrateCam(camIntrinsic4, camExtrinsic4, 'cam4')
    print('done ')
    img1 = camExtrinsic1.read()[1]
    img2 = camExtrinsic2.read()[1]
    img3 = camExtrinsic3.read()[1]
    img4 = camExtrinsic4.read()[1]

    drawCamPos(img1, cam1xml)
    drawCamPos(img2, cam2xml)
    drawCamPos(img3, cam3xml)
    drawCamPos(img4, cam4xml)


def calibrateCam(intrinsic, extrinsic, cam_string):
    print(cam_string)
    frames = getFrames(intrinsic, 20)
    ret, mtx, dist, rvecs, tvecs = Offline(frames)


    print(f'calibration {cam_string}:\n cam:\n{mtx},\n distortion:\n {dist}')

    ret, img = extrinsic.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = getChessboardCorners(gray)
    corners = interpolateCorners(corners, gray)

    # refine the corner positions
    corners2 = cv.cornerSubPix(gray, corners, (4, 4), (-1, -1), criteria)

    # Draw and display the corners
    cv.drawChessboardCorners(img, board_shape, corners2, ret)
    cv.imshow('img', img)
    cv.waitKey(0)


    ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

    R = cv.Rodrigues(rvecs)[0]
    T = -numpy.matrix(R).T * numpy.matrix(tvecs)

    #flip the y and z axis and negate z of T to get the correct translation
    T = np.matrix([[T[0, 0]], [-T[2, 0]], [T[1, 0]]])


    # rotate R by 180 degrees around y axis
    R = R * np.matrix([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    #rotete R by -90 degrees around z axis
    R = R * np.matrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    print(f'extrinsics {cam_string}:\n rotation:{rvecs},\n translation:\n {tvecs}, \n rotation matrix:\n {R}, \n translation matrix:\n {T}')

    camXML = cv.FileStorage(f'data/{cam_string}/config.xml', cv.FileStorage_WRITE)
    camXML.write('CameraMatrix', mtx)
    camXML.write('DistortionCoeffs', dist)
    camXML.write('rvec', rvecs)
    camXML.write('tvecs', tvecs)
    camXML.write('RotationMatrix', R)
    camXML.write('TranslationMatrix', T)


    cv.destroyAllWindows()
    camXML.release()

    return img

def drawCamPos(img, xml_file):
    mtx = xml_file.getNode('CameraMatrix').mat()
    dist = xml_file.getNode('DistortionMatrix').mat()
    # rvecs = xml_file.getNode('rvec').mat()
    tvecs = xml_file.getNode('tvecs').mat()
    R = xml_file.getNode('RotationMatrix').mat()
    T = xml_file.getNode('TranslationMatrix').mat()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # ret, corners = getChessboardCorners(gray)
    # corners = interpolateCorners(corners, gray)
    #
    # # refine the corner positions
    # corners2 = cv.cornerSubPix(gray, corners, (4, 4), (-1, -1), criteria)

    # Draw and display the corners
    # cv.drawChessboardCorners(img, board_shape, corners2, ret)
    # cv.imshow('img', img)
    # cv.waitKey(0)

    cam1pos = cam1xml.getNode('TranslationMatrix').mat().flatten()
    #negate the z axis
    cam1pos[2] = -cam1pos[2]
    print(cam1pos)
    cam2pos = cam2xml.getNode('TranslationMatrix').mat().flatten()
    cam2pos[2] = -cam2pos[2]
    cam3pos = cam3xml.getNode('TranslationMatrix').mat().flatten()
    cam3pos[2] = -cam3pos[2]
    cam4pos = cam4xml.getNode('TranslationMatrix').mat().flatten()
    cam4pos[2] = -cam4pos[2]
    cam_points = np.float32([cam1pos, cam2pos, cam3pos, cam4pos])
    print(cam_points)

    rvecs = cv.Rodrigues(R)[0]

    length = 4 * cube_size
    cube = np.float32([[0, 0, 0], [0, length, 0], [length, length, 0], [length, 0, 0],
                       [0, 0, -length], [0, length, -length], [length, length, -length], [length, 0, -length]])

    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(cube, rvecs, tvecs, mtx, dist)
    img = draw_cube(img, imgpts)

    campts, jac = cv.projectPoints(cam_points, rvecs, tvecs, mtx, dist)
    points = np.int32(campts).reshape(-1, 2)
    print(points)

    #for each point in points, draw a cam number
    for i in range(len(points)):
        img = cv.putText(img, str(i), tuple(points[i]), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

    cv.imshow('img', img)
    cv.waitKey(0)

    cv.destroyAllWindows()



if __name__ == "__main__":
    main()