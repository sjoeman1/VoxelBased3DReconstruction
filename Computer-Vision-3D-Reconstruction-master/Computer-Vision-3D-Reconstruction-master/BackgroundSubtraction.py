import cv2 as cv
import numpy as np


def compose_background(frames):
    #convert frames to HSV first
    for i in range(len(frames)):
        frames[i] = cv.cvtColor(frames[i], cv.COLOR_BGR2HSV)
    average = np.zeros(frames[0].shape, dtype=np.float32)
    #calculate standard deviation for hue, saturation and value
    sd = np.zeros(frames[0].shape, dtype=np.float32)
    #calculate average
    for frame in frames:
        average += frame
    average /= len(frames)
    #calculate standard deviation
    for frame in frames:
        sd += np.square(frame - average)
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

        #calculate difference
        diff = abs(hsv - background)

        cv.imshow('diff', diff)
        cv.waitKey(0)


        #check for each pixel if it is in the range of the standard deviation to create mask
        mask = create_mask(diff, sd)

        #refine mask
        img = refine_mask(mask)
        cv.imshow('mask', img)
        cv.waitKey(0)

        new_frames.append(img)
    return np.array(new_frames)

def refine_mask(mask):
    mask = cv.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=2)
    #find contours
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #draw the biggest contour
    c = max(contours, key=cv.contourArea)
    blank = np.zeros(mask.shape, dtype=np.uint8)
    img = cv.fillPoly(blank, [c], 255)
    return img

def create_mask(diff, sd):
    # check for each pixel for each channel if it is in the range of the standard deviation to create mask
    mask = np.zeros((diff.shape[0], diff.shape[1]), dtype=np.uint8)
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            standard_deviation = sd[i,j]
            #reflip bits
            for value in range(3):
                if standard_deviation[value] > 123:
                    standard_deviation[value] = 255-standard_deviation[value]

            d = diff[i,j]
            #reflip bits
            for value in range(3):
                if d[value] > 123:
                    d[value] = 255-d[value]
            sd_diff = np.abs(d - standard_deviation)

            #print(sd_diff)
            if (5 < sd_diff[0]) and (18 < sd_diff[1]) and (18 < sd_diff[2]):
                mask[i,j] = 255

    return mask

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

    cam_nr = 2
    background = cv.imread('data/cam' + str(cam_nr) +'/background.png')
    sd = cv.imread('data/cam' + str(cam_nr) +'/sd.png')
    video = cv.VideoCapture('data/cam' + str(cam_nr) +'/video.avi')
    new_frames = subtract_background(video, background, sd)
    cv.imshow('frame', new_frames[0])
    cv.waitKey(0)
if __name__ == '__main__':
        main()

