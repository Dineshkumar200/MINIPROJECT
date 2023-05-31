# <p align = "center"> FACE SHAPE AND HAIR STYLE MATCHING SYSTEM </p>

## ABSTRACT

## PROBLEM STATEMENT:
The problem at hand is to develop a hair style recommendation system utilizing face shape detection. The objective is to create an automated solution that can accurately determine the face shape of an individual based on their facial features and provide tailored hair style recommendations accordingly. 	

	The system should leverage computer vision techniques to analyze facial images, extract relevant facial landmarks, and apply an algorithm to accurately classify the face shape.                           Once the face shape is identified, the system should utilize a comprehensive database of hairstyles suitable for each face shape to suggest appropriate styles that complement the individual's features. 

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

