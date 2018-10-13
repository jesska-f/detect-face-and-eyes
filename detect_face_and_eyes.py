import cv2
import sys

# Get the path of the image that the user supplied
imagePath = sys.argv[1]

# Define the path to the image classifier files
faceCascPath = "haarcascade_frontalface_default.xml"
eyeCascPath = "haarcascade_eye.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(faceCascPath)
eyeCascade = cv2.CascadeClassifier(eyeCascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20),
    # flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    #For each face, locate the eyes, then draw a rectangle around them
    faceGray = gray[y:y+h, x:x+w]
    faceColor = image[y:y+h, x:x+w]
    eyes = eyeCascade.detectMultiScale(faceGray,scaleFactor=1.009,minNeighbors=8 )
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(faceColor,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

cv2.imshow("All around us are familiar faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
