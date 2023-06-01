# <p align = "center"> FACE SHAPE AND HAIR STYLE MATCHING SYSTEM </p>


## <br><br><br><br>PROBLEM STATEMENT
The problem at hand is to develop a hairstyle recommendation system based on face shape detection. The objective is to create an automated solution that can analyze the user's face shape and suggest suitable hairstyles that complement their unique features.

The primary challenge is to accurately detect and classify the user's face shape using computer vision techniques. Face shape detection involves analyzing facial landmarks, such as the position of the eyes, nose, and mouth, and determining the overall structure of the face. Once the face shape is identified, the system needs to provide personalized hairstyle recommendations that enhance the user's natural features, taking into account factors such as hair texture, length, and thickness.

Overall, the goal is to create a reliable and efficient hairstyle recommendation system that utilizes face shape detection technology to offer personalized and aesthetically pleasing hairstyle suggestions to users, promoting self-expression and confidence in their appearance.
  
## ARCHITECTURE DIAGRAM:

![Screenshot 2023-05-29 113117](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/5e2d748f-e107-4c5b-bf64-2922da598d77)

## METHODOLOGY:

### Detecting the Forehead Point using Bounding Box

In this step, we utilize dlib to perform face detection on the input image, which provides us with the bounding box coordinates of the detected face.

### Extracting Facial Features with Dlib

Once the face is detected, the subsequent task involves identifying the facial landmarks or key points on the face, including cheekbones, chin, jawline, and the highest point on the forehead.

### Calculating Distances between Extracted Points

Upon obtaining the facial landmarks, we can proceed to analyze the face shape by applying predefined criteria or rules. This analysis typically involves comparing the relative measurements of specific facial features, such as the forehead width, cheekbones, jawline, or the distance between certain points.

### Determining the Face Shape

Based on these measurements, we can classify the face shape into various categories, such as oval, round, square, heart-shaped, and more.

### Recommendations for Hairstyles based on Face Shape

Once the face shape is determined, we can employ a set of predefined rules or recommendations to suggest suitable hairstyles that complement the specific face shape.

## PROGRAM:


```python

# FACE SHAPE DETECTION

import cv2
import dlib
import numpy as np
import face_recognition
import math
import matplotlib.pyplot as plt

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the input image
image = cv2.imread('OIP (2).jfif')

# Detect the face landmarks in the input image
face_landmarks_list = face_recognition.face_landmarks(image)

# Plot the input image
plt.imshow(image)
ax = plt.gca()

# Loop over all the faces detected in the input image
for face_landmarks in face_landmarks_list:
    # Extract the chin points for this face
    chin_points = face_landmarks['chin']
    
    # Plot the chin points in unique colors
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, point in enumerate(chin_points):
        ax.scatter(point[0], point[1], c=colors[i%len(colors)], s=10)
        ax.annotate(str(i+1), xy=(point[0], point[1]), xytext=(point[0]+2, point[1]+2), color='black', fontsize=8)
    
# Show the plot
plt.show()

# FOREHEAD POINT

# Convert the image to grayscale (dlib face detector works on grayscale images)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray)

# Loop over the detected faces
for face in faces:
    # Extract the coordinates of the bounding box for each face
    x, y, w, h = face.left(), face.top(), face.width(), face.height()

    # Calculate the desired top point of the bounding box (e.g., where the hairline ends)
    hairline_y = int(y - 0.2 * h)  # Adjust the multiplier (0.2) as needed

    # Adjust the bounding box to include the forehead region
    h = int(h + (y - hairline_y))
    forehead_start= hairline_y
    
print(forehead_start)

# Detect the facial landmarks for the current face
landmarks = predictor(gray, face)

# Access the glabella point from the landmarks
glabella_point = (landmarks.part(27).x, landmarks.part(27).y)

# Create a figure and axis object
fig, ax = plt.subplots()

# Load the image
img = plt.imread("OIP (2).jfif")

# Display the image on the axis
ax.imshow(image)

# Plot a point with x=100 and y=200
ax.plot(glabella_point[0],forehead_start,'ro', alpha=1.0)

# Show the plot
plt.show()

# CHIN POINT DETECTION

# Access the 9th point of the chin for the first face detected in the input image
chin_9th_point = face_landmarks_list[0]['chin'][8]

# Print the coordinates of the 9th point of the chin
print(chin_9th_point)

chin_x = chin_9th_point[0]
chin_y = chin_9th_point[1]

# Create a figure and axis object
fig, ax = plt.subplots()

# Load the image
img = plt.imread("OIP (2).jfif")

# Display the image on the axis
ax.imshow(image)

# Plot a point with x=100 and y=200
ax.plot(chin_x, chin_y, 'ro', alpha=1.0)

# Show the plot
plt.show()

#  FACE LENGTH

# Calculate the distance between the forehead start point and chin point
facelength = abs(chin_9th_point[1] - forehead_start)

# Draw a line between the two points in lime color
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot([chin_9th_point[0],glabella_point[0] ], [chin_9th_point[1],forehead_start], color='lime', linewidth=2)

# Show the plot
plt.show()

face_x = chin_9th_point[0]
face_y = chin_9th_point[1]

face1_y = forehead_start
face1_x = glabella_point[0]

# Create a figure and axis object
fig, ax = plt.subplots()

# Display the image on the axis
ax.imshow(image)

# Plot a point with x=100 and y=200
ax.plot(face_x , face_y ,'ro', alpha=1.0)
ax.plot(face1_x , face1_y ,'ro', alpha=1.0)

# Show the plot
plt.show()
#Display the jawline distance
print(facelength)


# Print the distance
print("Distance between forehead start and chin point:", facelength)

# FOREHEAD DISTANCE

# Extract the facial landmarks for the first face detected
landmarks = predictor(gray, face)

# Detect the face landmarks in the input image
faces = detector(image)
landmarks = predictor(image, faces[0])

# Define the left eyebrow landmark indices
left_eyebrow_indices = [17, 18, 19, 20, 21]

# Define a color map for each landmark
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Plot the landmarks on the image
for i, idx in enumerate(left_eyebrow_indices):
    x = landmarks.part(idx).x
    y = landmarks.part(idx).y
    plt.scatter(x, y, color=colors[i], s=50)

# Display the image with landmarks
plt.imshow(image[:, :, ::-1])
plt.show()

# Define the eyebrow landmark indices
left_eyebrow_indices = [17]
right_eyebrow_indices = [25]

# Define a color map for each landmark
colors = ['red', 'blue']

# Plot the landmarks on the image
for i, idx in enumerate(left_eyebrow_indices + right_eyebrow_indices):
    x = landmarks.part(idx).x
    y = landmarks.part(idx).y
    plt.scatter(x, y, color=colors[i], s=50)

# Display the image with landmarks
plt.imshow(image[:, :, ::-1])
plt.show()

# Calculate the distance between the left and right eyebrow landmarks
left_x, left_y = landmarks.part(17).x, landmarks.part(17).y
right_x, right_y = landmarks.part(25).x, landmarks.part(25).y
forehead = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)


# Calculate the distance between the left and right eyebrow landmarks
left_x, left_y = landmarks.part(17).x, landmarks.part(17).y
right_x, right_y = landmarks.part(25).x, landmarks.part(25).y
distance = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)

# Draw a lime line between the left and right eyebrow landmarks
plt.plot([left_x, right_x], [left_y, right_y], color='lime', linewidth=2)

# Display the image with landmarks and the distance between the eyebrow landmarks
plt.title(f'Distance between landmarks: {distance:.2f}')
plt.imshow(image[:, :, ::-1])
plt.show()


# Display the forehead distance
print("Forehead distance:", forehead)


# JAWLINE DISTANCE


# Get the coordinates of the 10th and 13th Facial points
point_10 = face_landmarks_list[0]['chin'][9]
point_14 = face_landmarks_list[0]['chin'][12]

# Compute the Euclidean distance between the two points
jawlength = ((point_10[0]-point_14[0])**2 + (point_10[1]-point_14[1])**2)**0.5

# Draw a line between the two points in lime color
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot([point_10[0], point_14[0]], [point_10[1], point_14[1]], color='lime', linewidth=2)

# Show the plot
plt.show()

jaw_x = point_10[0]
jaw_y = point_10[1]

jaw1_x = point_14[0]
jaw1_y = point_14[1]

# Create a figure and axis object
fig, ax = plt.subplots()

# Display the image on the axis
ax.imshow(image)

# Plot a point with x=100 and y=200
ax.plot(jaw_x , jaw_y ,'ro', alpha=1.0)
ax.plot(jaw1_x , jaw1_y ,'ro', alpha=1.0)

# Show the plot
plt.show()
#Display the jawline distance
print(jawlength)

# CHIN DISTANCE

# Extract the facial landmarks for the first face in the input image
chin_points = face_landmarks_list[0]['chin']

# Get the 8th and 10th points of the chin
point_8 = chin_points[7]
point_10 = chin_points[9]

# Compute the Euclidean distance between the 8th and 10th points of the chin
chinlength = math.sqrt((point_8[0]-point_10[0])**2 + (point_8[1]-point_10[1])**2)

# Draw a line between the two points in lime color
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot([point_8[0], point_10[0]], [point_8[1], point_10[1]], color='lime', linewidth=2)

# Show the plot
plt.show()

chinln_x = point_8[0]
chinln_y = point_8[1]

chinln1_x = point_10[0]
chinln1_y = point_10[1]

# Create a figure and axis object
fig, ax = plt.subplots()

# Display the image on the axis
ax.imshow(image)

# Plot a point with x=100 and y=200
ax.plot(chinln_x , chinln_y ,'ro', alpha=1.0)
ax.plot(chinln1_x , chinln1_y ,'ro', alpha=1.0)

# Show the plot
plt.show()
#Display the jawline distance
print(chinlength)


# Print the distance between the 8th and 10th points of the chin
print("Distance between 8th and 10th points of the chin:", chinlength)

# CHEEK BONE DISTANCE

#Extract the facial landmarks for the first face in the input image
chin_points = face_landmarks_list[0]['chin']

# Get the 2th and 16th points of the chin
point_2 = chin_points[1]
point_16 = chin_points[15]

# Compute the Euclidean distance between the 2th and 16th points of the chin
cheekbone = math.sqrt((point_2[0]-point_16[0])**2 + (point_2[1]-point_16[1])**2)

# Draw a line between the two points in lime color
fig, ax = plt.subplots()
ax.imshow(image)
ax.plot([point_2[0], point_16[0]], [point_2[1], point_16[1]], color='lime', linewidth=2)

# Show the plot
plt.show()

# Print the distance between the 2th and 16th points of the chin
print("Distance between 2th and 16th points of the Facial landmarks:", cheekbone)

# FACE SHAPE DETECTING FUNCTION

def calculate_ratios(facelength, cheekbone, forehead, jawlength, chinlength):
    cheekbone_ratio = cheekbone / facelength
    jawline_ratio = jawlength / facelength
    forehead_ratio = forehead / facelength
    chin_ratio = chinlength / facelength

    return cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio

def determine_face_shape(cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio):
    if forehead_ratio > 0.27:
        if chin_ratio < 0.18:
            if cheekbone_ratio > 0.51:
                return "Heart-shaped"
            else:
                return "Oval"
        else:
            if cheekbone_ratio > 0.47 and jawline_ratio > 0.37:
                return "Square"
            elif cheekbone_ratio > 0.47 and jawline_ratio < 0.36:
                return "Diamond"
            else:
                return "Triangle"
    else:
        if cheekbone_ratio < 0.48:
            if jawline_ratio < 0.36:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    return "Soft"
            else:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    return "Square"
        else:
            if jawline_ratio < 0.36:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    return "Soft"
            else:
                if chin_ratio < 0.20:
                    return "Round"
                else:
                    if forehead_ratio > 0.50:
                        return "Oblong"
                    else:
                        return "Square"

cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio = calculate_ratios(facelength, cheekbone, forehead, jawlength, chinlength)

face_shape = determine_face_shape(cheekbone_ratio, jawline_ratio, forehead_ratio, chin_ratio)
print("Face shape:", face_shape)

def suggest_hairstyle(face_shape):
    # Define hairstyle suggestions and things to avoid based on face shape
    hairstyles = {
        "Oval": {
            "suggestions": ["Layered cuts", "Bob hairstyles", "Side-swept bangs"],
            "avoid": ["Heavy bangs", "Excessive volume on top"]
        },
        "Round": {
            "suggestions": ["Longer hairstyles", "Angled bobs", "Textured layers"],
            "avoid": ["Full, straight bangs", "Chin-length cuts"]
        },
        "Square": {
            "suggestions": ["Soft, layered cuts", "Side-parted styles", "Curly or wavy hair"],
            "avoid": ["Straight, blunt cuts", "Center parts"]
        },
        "Heart-shaped": {
            "suggestions": ["Pixie cuts", "Long, wavy styles", "Side-swept bangs"],
            "avoid": ["Short, tight curls", "Heavy, straight bangs"]
        },
        "Diamond": {
            "suggestions": ["Chin-length bobs", "Shoulder-length styles", "Side-swept bangs"],
            "avoid": ["Excessive volume on sides", "Short, cropped styles"]
        },
        "Triangle": {
            "suggestions": ["Layered cuts with volume on top", "Textured pixie cuts"],
            "avoid": ["Excessive volume on the sides", "Very short hairstyles"]
        },
        "Oblong": {
            "suggestions": ["Medium-length layered cuts", "Fringes or bangs", "Soft waves"],
            "avoid": ["Very long, straight hair", "High ponytails"]
        }
    }
    
    # Check if the given face shape is in the dictionary
    if face_shape in hairstyles:
        suggestion = hairstyles[face_shape]["suggestions"]
        avoid = hairstyles[face_shape]["avoid"]
        return f"For your {face_shape} face shape, you can try hairstyles like {', '.join(suggestion)}. Avoid {', '.join(avoid)}."

    # If the face shape is not recognized
    return "Sorry, we don't have specific hairstyle recommendations for that face shape."
    
hairstyle_suggestion = suggest_hairstyle(face_shape)
print(hairstyle_suggestion)




```

## OUTPUT:


### FACIAL LANDMARKS:
![Screenshot 2023-06-01 104347](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/20726ea5-d98e-4a65-b5ba-c6bfae844962)

### FOREHEAD POINT:
![Screenshot 2023-06-01 104403](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/a647637a-7ac6-4e72-9b7e-1a0ad3f06e41)

### CHIN POINT:
![Screenshot 2023-06-01 104416](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/65e47ebb-2f15-4f2f-8f8b-cf48ba7acbd2)


### <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>FACE LENGTH:
![Screenshot 2023-06-01 104431](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/98232274-c2c8-40ae-8125-2d30259d2c72)
### <br><br><br><br><br><br><br><br><br><br>FOREHEAD DISTANCE:

![Screenshot 2023-06-01 120306](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/c15e08cd-60e8-4968-8d6e-f4b867dca38f)


### <br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>JAWLINE DISTANCE:
<br><br>
![Screenshot 2023-06-01 104531](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/a120199d-d142-4a67-bd9d-d92dd94552ba)

### <br><br><br><br><br><br><br>CHIN DISTANCE:
<br><br>
![Screenshot 2023-06-01 104545](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/0c72b319-47b7-472f-80f9-c2b71a2e8030)

### <br><br><br><br><br><br><br><br><br><br><br>CHEEK BONE DISTANCE:
![Screenshot 2023-06-01 104600](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/bebe5bbe-0475-4f11-a19a-08f9d674f000)

### FACE SHAPE DETECTION:
![Screenshot 2023-06-01 104620](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/b7a21091-5c28-46e6-8b21-3da60b5b344f)


### HAIRSTYLE RECOMMMENDATION:
![Screenshot 2023-06-01 104636](https://github.com/Dineshkumar200/MINIPROJECT/assets/75235789/4f422f03-3c7c-4baf-9a95-35883b4d6776)


## <br><br><br><br>CONCLUSION:

Facial shape detection and analysis for hairstyle recommendation systems holds great potential for providing personalized and accurate hairstyle recommendations to users.
By utilizing computer vision and machine learning techniques, facial landmarks and geometric features can be extracted to accurately analyze the user's facial shape. 
Ultimately, the use of facial shape detection and hairstyle recommendation systems can provide a more efficient and satisfying experience for individuals looking to change their hairstyle.


