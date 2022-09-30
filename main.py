import os
import cv2
import numpy as np


def opt_one_img(frame, frame_pre, view_pre):
    diff = abs((np.linalg.norm(frame) - np.linalg.norm(frame_pre))) / np.linalg.norm(frame)
    print(np.linalg.norm(frame - frame_pre), diff)
    if diff <= 1:
        view = view_pre
    else:
        if view_pre == 'view1':
            view = 'view2'
        elif view_pre == 'view2':
            view = 'view1'
    view_pre = view
    return view, view_pre


if __name__ == '__main__':
    videoPath = 'E:/pythonWorkspace/dataWorkspace/scu'
    videoName = '川大红外数据.mp4'
    path = os.path.join(videoPath, videoName)
    capture = cv2.VideoCapture(path)
    frame_base_path = videoPath + '/frame/8bit/'
    frame_pre = []
    view_pre = 'view1'
    if capture.isOpened():
        frameNum = 0
        while True:
            ret, frame = capture.read()
            if not ret:
                break
            frame = frame[:, :, 0]
            if frameNum == 0:
                frame_pre = np.full(frame.shape, 0)
            cv2.imshow('frame', frame)
            frameNum += 1
            print(frameNum)
            cv2.waitKey(1)
            view, view_pre = opt_one_img(frame, frame_pre, view_pre)  # 自己写函数op_one_img()逐帧处理
            frame_path = os.path.join(frame_base_path, view)
            cv2.imwrite(frame_path + '/frame_' + str(frameNum) + '.jpg', frame)
            frame_pre = frame
    else:
        print('视频打开失败！')
