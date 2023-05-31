# <p align = "center"> FACE SHAPE AND HAIR STYLE MATCHING SYSTEM </p>

## ABSTRACT

## PROBLEM STATEMENT
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

