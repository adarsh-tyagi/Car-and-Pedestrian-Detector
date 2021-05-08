import cv2

car_classifier = cv2.CascadeClassifier("cars.xml")
pedestrian_classifier = cv2.CascadeClassifier("pedestrian.xml")

# detection in image
img = cv2.imread("image3.jpg")
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

points_car = car_classifier.detectMultiScale(grayscale_img)
print("Car points: ", points_car)

points_pedestrian = pedestrian_classifier.detectMultiScale(grayscale_img)
print("Pedestrian points: ", points_pedestrian)

for (x, y, w, h) in points_car:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

for (x, y, w, h) in points_pedestrian:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

cv2.imshow("Car and Pedestrian Detector", img)
cv2.waitKey()

# detection in video
video = cv2.VideoCapture("video2.mp4")
while True:
    success, frame = video.read()
    if success:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        points_car = car_classifier.detectMultiScale(grayscale_frame)
        points_pedestrian = pedestrian_classifier.detectMultiScale(grayscale_frame)

        for (x, y, w, h) in points_car:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)

        for (x, y, w, h) in points_pedestrian:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    else:
        break

    cv2.imshow("Car and Pedestrian Detector", frame)
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()

print("Program ran successfully")

