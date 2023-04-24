# Feature: 
### is a piece of information which is relevant for solving the computational task related to a certain application. Features may be specific structures in the image such as points, edges or objects.

### - The features can be classified into two main categories:
1. ### The features that are in specific locations of the images, such as mountain peaks, building corners, doorways, or interestingly shaped patches of snow. These kinds of localized features are often called __keypoint features__.
2. ### The features that can be matched based on their orientation and local appearance (edge profiles) are called __edges__.

<br />

## - Feature Detection:
- ### Detection: Identify the Interest Point.
- ### Description: The local appearance around each feature point is described in some way that is (ideally) invariant under changes in illumination, translation, scale, and in-plane rotation.
- ### Matching: Descriptors are compared across the images, to identify similar features. For two images we may get a set of pairs (Xi, Yi) ↔ (Xi´, Yi´), where (Xi, Yi) is a feature in one image and (Xi´, Yi´) its matching feature in the other image.

- ### Key points: Key points, also known as interest points, They are spatial locations, or points in the image that define what is interesting or what stand out in the image.
<br />

### - Algorithms for Identification:
1. ### Harris Corner.
2. ### SIFT(Scale Invariant Feature Transform).
3. ### SURF(Speeded Up Robust Feature).
4. ### FAST(Features from Accelerated Segment Test).
5. ### ORB(Oriented FAST and Rotated BRIEF).


## - Feature Descriptor
### A feature descriptor is an algorithm which takes an image and outputs feature descriptors/feature vectors. Feature descriptors encode interesting information into a series of numbers and act as a sort of numerical “fingerprint” that can be used to differentiate one feature from another.

### - Local Descriptor: It is a compact representation of a point’s local neighborhood. Local descriptors try to resemble shape and appearance only in a local neighborhood around a point and thus are very suitable for representing it in terms of matching.
### - Global Descriptor: A global descriptor describes the whole image. They are generally not very robust as a change in part of the image may cause it to fail as it will affect the resulting descriptor.

<br />

## - Algorithms:
1. ### SIFT(Scale Invariant Feature Transform)
2. ### SURF(Speeded Up Robust Feature)
3. ### BRISK (Binary Robust Invariant Scalable Keypoints)
4. ### BRIEF (Binary Robust Independent Elementary Features)
5. ### ORB(Oriented FAST and Rotated BRIEF)