# -*- coding:utf-8 -*-
import cv2

from train import Model
from image import IMAGE_CHANNELS

class FaceResult:
  def __init__(self, name, rect, color):
    self.name = name
    self.rect = rect
    self.color = color


if __name__ == '__main__':
    #cascade_path = "/usr/local/opt/opencv3/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
    cascade_path = "/usr/local/opt/opencv3/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml"
    model = Model()
    model.load()

    cap = cv2.VideoCapture(0)
    windowName = 'face'

    cascade = cv2.CascadeClassifier(cascade_path)

    while True:
        ret, frame = cap.read()
        if ret == False:
            print('failed to read')
            continue

        waitTime = 10

        # グレースケール化
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 顔認識
        facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(64, 64))
        face_results = []
        if len(facerect) > 0:
            detected_color = (255, 255, 255)
            unknown_color = (255, 64, 64)
            for rect in facerect:
                x, y = rect[0:2]
                width, height = rect[2:4]
                if IMAGE_CHANNELS == 1:
                    image = frame_gray[y: y + height, x: x + width]
                else:
                    image = frame[y: y + height, x: x + width]

                threshould = 0.8
                detected = False
                fr = FaceResult(None, rect, unknown_color)
                result = model.predict(image)
                if result[0][0] > threshould:  # su-metal
                    fr.name = 'SU-METAL'
                elif result[0][1] > threshould:  # yui-metal
                    fr.name = 'YUIMETAL'
                elif result[0][2] > threshould:  # moa-metal
                    fr.name = 'MOAMETAL'

                if fr.name != None:
                    fr.color = detected_color
                    #waitTime = 500

                face_results.append( fr )

        # 判定領域を描画
        if len(face_results) > 0:
            for fr in face_results:
                cv2.rectangle(frame, tuple(fr.rect[0:2]), tuple(fr.rect[0:2] + fr.rect[2:4]), fr.color, thickness=2)
                cv2.putText(frame, fr.name, (fr.rect[0], fr.rect[1] + fr.rect[3] ), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fr.color)

        cv2.imshow(windowName, frame)

        k = cv2.waitKey(waitTime)
        #Escキーを押されたら終了
        if k == 27:
            break

    #キャプチャを終了
    cap.release()
    cv2.destroyAllWindows()


