import cv2
import base64


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # set buffer to 1 frame (model needs to get the last frame)

def get_frame():

    for i in range(3): # clean the buffer
        cap.read()

    ret, frame = cap.read()
    if not ret:
        print("Could not get the frame")
        return None

    frame = cv2.resize(frame, (640, 480))

    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    b64_image = base64.b64encode(buffer).decode('utf-8')

    return b64_image

    if __name__ == "__main__":
        frame =get_frame()

    if frame:
        print(f"got it.")
        
    cap.release()