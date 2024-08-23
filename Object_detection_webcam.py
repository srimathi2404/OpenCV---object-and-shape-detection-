import cv2
import numpy as np


bounding_box = (100, 100, 300, 300)  # Modify this if wanted

def detect_shapes(image):
    # Extract the region of interest 
    x, y, w, h = bounding_box
    roi = image[y:y+h, x:x+w]
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
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
            (rx, ry, rw, rh) = cv2.boundingRect(approx)
            ar = rw / float(rh)
            if 0.95 <= ar <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif len(approx) > 4:
            area = cv2.contourArea(contour)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity > 0.9: 
                shape = "Circle"
        
   
        if shape is not None:
            cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)
            (rx, ry, rw, rh) = cv2.boundingRect(contour)
            cv2.putText(roi, shape, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.rectangle(roi, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
    
   
    image[y:y+h, x:x+w] = roi
    
    return image

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video device")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        
        x, y, w, h = bounding_box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
       
        detected_shapes = detect_shapes(frame)
        

        cv2.imshow("Shape Detection", detected_shapes)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
