import cv2
import math

input_video_path = 'sample.mp4'

cap = cv2.VideoCapture(input_video_path)


def process_image(image):

    im_h, im_w, _ = image.shape

    out = image

    res = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(res, (15, 15), 0)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        #Detecting center of object 

        distance = math.floor(((im_h // 2 - (y + h // 2)) ** 2 + (im_w // 2 - (x + w // 2)) ** 2) ** 0.5)
        pad = 10
        cv2.putText(image, str(distance) + 'px', (50, 50), cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255))
        cv2.line(image, (im_w // 2, im_h // 2), (x + w // 2, y  + h // 2), (0, 0, 255), 1)

    return out


while True:
    ret, frame = cap.read()



    if not ret:
        break

    image_to_show  = process_image(frame) 
    cv2.imshow('frame', image_to_show)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

