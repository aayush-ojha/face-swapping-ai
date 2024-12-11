import cv2
import dlib
import numpy as np


face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def get_landmarks(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    if len(faces) > 0:
        return landmark_predictor(gray, faces[0])
    return None


def swap_images(img1, img2):
    return img1, img2


def align_faces(img, landmarks):

    left_eye = (landmarks.part(36).x, landmarks.part(36).y)
    right_eye = (landmarks.part(45).x, landmarks.part(45).y)


    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))


    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    
    (h, w) = img.shape[:2]
    aligned_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    return aligned_img


def warp_images(img1, img2, landmarks1, landmarks2):
    # Get facial landmarks as points
    points2 = [(landmarks2.part(i).x, landmarks2.part(i).y) for i in range(68)]  
    points1 = [(landmarks1.part(i).x, landmarks1.part(i).y) for i in range(68)]  

    points2 = np.array(points2, dtype=np.float32)
    points1 = np.array(points1, dtype=np.float32)


    M, _ = cv2.findHomography(points2, points1, cv2.RANSAC, 5.0)


    warped_img2 = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))

    return warped_img2, img1  


def blend_images(img1, img2, landmarks1, landmarks2):

    mask = np.zeros_like(img1, dtype=np.uint8)
    

    points = np.array([(landmarks1.part(i).x, landmarks1.part(i).y) for i in range(68)])
    hull = cv2.convexHull(points)
    mask = cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    

    center = (np.mean(points[:, 0]).astype(int), np.mean(points[:, 1]).astype(int))
    

    output = cv2.seamlessClone(
        img2, img1, mask, center, cv2.NORMAL_CLONE
    )
    
    return output

if __name__ == "__main__":

    img1_path = input("Enter the path for the first image: ")
    img2_path = input("Enter the path for the second image: ")


    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        print(f"Error: Could not load image from {img1_path}")
        exit(1)
    if img2 is None:
        print(f"Error: Could not load image from {img2_path}")
        exit(1)

    img1, img2 = swap_images(img1, img2)
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)

    if landmarks1 is not None and landmarks2 is not None:
        
        img2_aligned = align_faces(img2, landmarks2)  
        img1_aligned = align_faces(img1, landmarks1)  


        warped_face, target_img = warp_images(img1_aligned, img2_aligned, landmarks1, landmarks2)


        result = blend_images(target_img, warped_face, landmarks1, landmarks2)

        cv2.imwrite('result.jpg', result)
        print("Face swap completed successfully!")
    else:
        print("Could not detect faces in one or both images.")

