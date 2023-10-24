import cv2
import sys
import os.path

CASCADE_FILE = "./rsc/csc.xml"

def detect(filename):
    if not os.path.isfile(CASCADE_FILE):
        raise RuntimeError("%s: not found" % CASCADE_FILE)

    cascade = cv2.CascadeClassifier(CASCADE_FILE)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    
    faces = cascade.detectMultiScale(gray,
                                     scaleFactor = 1.05, # Up to 1.4 for speed
                                     minNeighbors = 5,   # 3 ~ 6
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)

# Check if there is a matching amount of arguments
if len(sys.argv) != 2:
    sys.stderr.write("usage: main.py <filename>\n")
    sys.exit(-1)
    
detect(sys.argv[1])