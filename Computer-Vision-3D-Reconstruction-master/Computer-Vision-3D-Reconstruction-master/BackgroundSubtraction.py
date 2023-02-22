import cv2 as cv
import numpy as np


def compose_background(frames):
    average = np.zeros(frames[0].shape, dtype=np.float32)
    sd = np.zeros(frames[0].shape, dtype=np.float32)
    for frame in frames:
        average += frame
        sd += (frame - average)**2
    average /= len(frames)
    sd = np.sqrt(sd / len(frames))
    return np.uint8(average), np.uint8(sd)


def subtract_background(video, background, sd):
    new_frames = []
    #convert frames to HSV
    ret = True
    while ret:
        ret, frame = video.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        cv.imshow('frame', hsv)
        cv.waitKey(1)
        hsv_background = cv.cvtColor(background, cv.COLOR_BGR2HSV)
        hsv_sd = cv.cvtColor(sd, cv.COLOR_BGR2HSV)

        #calculate difference
        diff = np.clip(hsv - hsv_background, 0, 255)
        cv.imshow('diff', diff)
        cv.waitKey(0)

        # calculate lower and upper bounds for each pixel
        lower = np.zeros(diff.shape, dtype=np.uint8)
        upper = hsv_sd
        #check for each pixel if it is in the range of the standard deviation to create mask
        mask = cv.inRange(diff, lower, upper)
        cv.imshow('mask', mask)
        cv.waitKey(0)

        res = cv.bitwise_and(frame, frame, mask= mask)
        # cv.imshow('res', res)
        # cv.waitKey(0)

        new_frames.append(res)
    return np.array(new_frames)

def getFrames(cam, num_frames= 30):
    frames = []
    totalFrames = cam.get(cv.CAP_PROP_FRAME_COUNT)
    _, img = cam.read()
    for i in np.linspace(0, totalFrames, num= int(num_frames), dtype= int, endpoint=False):
        cam.set(cv.CAP_PROP_POS_FRAMES, i)
        _, img = cam.read()
        frames.append(img)
    return np.array(frames)

def main():
    # for i in range(4):
    #     video = cv.VideoCapture('data/cam' + str(i+1) + '/background.avi')
    #     frames = getFrames(video)
    #     background, sd = compose_background(frames)
    #     cv.imwrite('data/cam' +  str(i+1) + '/background.png', background)
    #     cv.imwrite('data/cam' +  str(i+1) + '/sd.png', sd)

    background = cv.imread('data/cam2/background.png')

    sd = cv.imread('data/cam2/sd.png')
    video = cv.VideoCapture('data/cam2/video.avi')
    new_frames = subtract_background(video, background, sd)
    cv.imshow('frame', new_frames[0])
    cv.waitKey(0)
if __name__ == '__main__':
        main()

