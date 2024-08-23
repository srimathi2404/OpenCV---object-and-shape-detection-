import cv2
import numpy as np

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True) 
        
        shape = None
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
            if 0.95 <= ar <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif len(approx) > 4:
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.9:  # High threshold for circularity to detect circles
                shape = "Circle"
        
    
        if shape is not None:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.putText(image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    return image

# Load the image
image_path = 'Path_to_your_image.jpg'  # Replace with your local image path
image = cv2.imread(image_path)

detected_shapes = detect_shapes(image)

# Display the output using cv2.imshow
cv2.imshow('Detected Shapes', detected_shapes)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
