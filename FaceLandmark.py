#FaceLandmarkingModuleCodeSnippet
from imutils import face_utils
import dlib
import cv2
 
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
 
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
    
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()


#get_frontal_face_detector__and__shape_predictor__CodeSnippet
import time
import dlib
from utils import applyAffineTransform, rectContains, calculateDelaunayTriangles, warpTriangle, face_swap3

if __name__ == '__main__' :

    # Make sure OpenCV is version 3.0 or above
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) &lt; 3 :
        print &gt;&gt;sys.stderr, 'ERROR: Script needs OpenCV 3.0 or higher'
        sys.exit(1)

    print("[INFO] loading facial landmark predictor...")
    model = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model)

    # Read images will swap image1 into image2
    filename1 = 'ted_cruz.jpg'
    filename1 = 'brad.jpg'
    #filename1 = 'hillary_clinton.jpg'

    img1 = cv2.imread(filename1);
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)
    shape1 = predictor(gray1, rects1[0])
    points1 = face_utils.shape_to_np(shape1) #type is a array of arrays (list of lists)
    #need to convert to a list of tuples
    points1 = list(map(tuple, points1))
