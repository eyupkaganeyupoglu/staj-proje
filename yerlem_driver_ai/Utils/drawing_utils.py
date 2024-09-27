# import cv2

class DrawingUtils:
    def __init__(self, mp_drawing, mp_drawing_styles, mp_face_mesh): #, mp_hands
        # DESCRIPTION: Initialize MediaPipe drawing modules from the MediaPipeUtils class.
        self.mp_drawing = mp_drawing
        self.mp_drawing_styles = mp_drawing_styles
        self.mp_face_mesh = mp_face_mesh
        # self.mp_hands = mp_hands

    # DESCRIPTION: Draw face landmarks on the image.
    def draw_face_landmarks(self, image, face_results): #, points_of_interest):
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                try:
                    # DESCRIPTION: Draw lips.
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LIPS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    # DESCRIPTION: Draw left eye.
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    # DESCRIPTION: Draw left eyebrow.
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=landmarks,
                        connections=self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    # DESCRIPTION: Draw right eye.
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=landmarks,
                        connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    # DESCRIPTION: Draw right eyebrow.
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=landmarks,
                        connections=self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style())
                    # DESCRIPTION: Draw spesific points on the face.
                    # for i, landmark in enumerate(landmarks.landmark):
                    #     if i in [point for points in points_of_interest.values() for point in points]:
                    #         x = int(landmark.x * image.shape[1])
                    #         y = int(landmark.y * image.shape[0])
                    #         cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                except Exception as e:
                    print(f"Error drawing face landmarks: {e}")

    # DESCRIPTION: Draw hand landmarks on the image.
    # def draw_hand_landmarks(self, image, hand_results):
    #     if hand_results.multi_hand_landmarks:
    #         for landmarks in hand_results.multi_hand_landmarks:
    #             try:
    #                 # DESCRIPTION: Draw hand.
    #                 self.mp_drawing.draw_landmarks(
    #                     image=image,
    #                     landmark_list=landmarks,
    #                     connections=self.mp_hands.HAND_CONNECTIONS,
    #                     landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
    #                     connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style())
    #             except Exception as e:
    #                 print(f"Error HAND_CONNECTIONS: {e}")