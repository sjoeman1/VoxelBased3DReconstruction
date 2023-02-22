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

CB_data = cv.FileStorage('data\checkerboard.xml', cv.FileStorage_READ)

s = cv.FileStorage('data\checkerboard.xml', cv.FileStorage_READ)

#chessboard parameters
columns = int(s.getNode('CheckerBoardWidth').real())
rows = int(s.getNode('CheckerBoardHeight').real())
board_shape = (columns, rows)
cube_size = int(s.getNode('CheckerBoardSquareSize').real())
s.release()

# prepare object points with cube size, like (0,0,0), (22,0,0), (44,0,0) ....,(132,110,0)
objp = np.zeros((columns * rows, 3), np.float32)
objp[:, :2] = cube_size * np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)

#amount of clicks for the corners
clicks = 0

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
        ret, corners = cv.findChessboardCorners(gray, board_shape, None)
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
        cv.drawChessboardCorners(img, board_shape, corners, ret)
        cv.imshow('img', img)

    cv.destroyWindow('img')

    #calibrate the camera using all the points found in all the images
    calibration = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return calibration

def getFrames(cam):
    frames = []
    totalFrames = cam.get(cv.CAP_PROP_FRAME_COUNT)
    print(totalFrames)
    _, img = cam.read()
    for i in numpy.linspace(0, totalFrames, num= 10, dtype= int, endpoint=False):
        cam.set(cv.CAP_PROP_POS_FRAMES, i)
        ret = False
        while not ret:
            _, img = cam.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            # Find the chess board corners automatically
            ret, corners = cv.findChessboardCorners(gray, board_shape, None)
        if ret:
            print(cam.get(cv.CAP_PROP_POS_FRAMES))
            frames.append(img)
    return np.array(frames)



def main():
    #TODO get frames from each cam
    calibrateCam(camIntrinsic1, camExtrinsic1, 'cam1')
    calibrateCam(camIntrinsic2, camExtrinsic2, 'cam2')
    calibrateCam(camIntrinsic3, camExtrinsic3, 'cam3')
    calibrateCam(camIntrinsic4, camExtrinsic4, 'cam4')
    print('done ')
    #TODO calibrate each camera using these frames


    #TODO write calibration to XML (manually?)


def calibrateCam(intrinsic, extrinsic, cam_string):
    print(cam_string)
    frames = getFrames(intrinsic)
    ret, mtx, dist, rvecs, tvecs = Offline(frames)
    print(f'calibration {cam_string}: cam:{mtx},\n distortion: {dist}')

    ret, img = extrinsic.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = getChessboardCorners(gray)
    corners = interpolateCorners(corners, gray)

    # refine the corner positions
    corners2 = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # Draw and display the corners
    cv.drawChessboardCorners(img, board_shape, corners, ret)
    cv.imshow('img', img)
    cv.waitKey(0)


    ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

    print(f'extrinsics {cam_string}: rotation:{rvecs},\n translation: {tvecs}')

    camXML = cv.FileStorage(f'data/{cam_string}/intrinsics.xml', cv.FileStorage_WRITE)
    camXML.write('CameraMatrix', mtx)
    camXML.write('DistortionCoeffs', dist)
    camXML.write('RotationMatrix', rvecs)
    camXML.write('TranslationMatrix', tvecs)

    cv.destroyAllWindows()
    camXML.release()


if __name__ == "__main__":
    main()