import cv2
import numpy as np
import subprocess
import sys

def start_menu():
    while True:
        frame = np.ones((500,1000,3), dtype=np.uint8)*255

        cv2.putText(frame,"SIGN LANGUAGE SYSTEM",
                    (220,100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

        # English button
        cv2.rectangle(frame,(150,200),(400,300),(0,255,0),-1)
        cv2.putText(frame,"ENGLISH",(200,260),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

        # Hindi button
        cv2.rectangle(frame,(600,200),(850,300),(255,140,0),-1)
        cv2.putText(frame,"HINDI",(670,260),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

        cv2.imshow("Start Menu",frame)
        key = cv2.waitKey(1)

        if key == ord('e'):
            cv2.destroyAllWindows()
            subprocess.call([sys.executable, "4_predict.py", "EN"])

        if key == ord('h'):
            cv2.destroyAllWindows()
            subprocess.call([sys.executable, "4_predict.py", "HI"])

        if key == 27:
            break

start_menu()
