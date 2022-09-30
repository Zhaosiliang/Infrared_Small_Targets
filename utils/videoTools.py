import glob

import cv2


def readVideo(path):
    capture = cv2.VideoCapture(path)
    if capture.isOpened():
        while True:
            ret, frame = capture.read()  # img 就是一帧图片
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if not ret:
                break  # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')


# def writeVideo(outputPath):
#     videoname = 'videoname_out.avi'
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     writer = cv2.VideoWriter(outputPath, fourcc, 1.0, (1280, 960), True)
#     imgpaths = glob.glob('*.jpg')
#     for path in imgpaths:
#         print(path)
#         img = cv2.imread(path)
#         writer.write(img)  # 读取图片后一帧帧写入到视频中
#     writer.release()


def opt_video():
    videoinpath = 'videoname.avi'
    capture = cv2.VideoCapture(videoinpath)
    if capture.isOpened():
        while True:
            ret, img_src = capture.read()
            if not ret:
                break
            img_out = opt_one_img(img_src)  # 自己写函数op_one_img()逐帧处理
    else:
        print('视频打开失败！')


def opt_one_img():
    pass
