import cv2
import face_recognition
import numpy as np
import tensorflow as tf
from PIL import Image


class DetectEmotion(object):
    def __init__(self, path_model_emotion_classification='./weight/model.h5'):
        self.mapper = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        self.path_model_emotion_classification = path_model_emotion_classification
        self.model_emotion_classification = tf.keras.models.load_model(self.path_model_emotion_classification)
        self.model_emotion_classification.summary()

    @staticmethod
    def draw_result(img, dict_res):
        img_save = img.copy()
        for location, label in dict_res:
            # print(location, label)
            top, right, bottom, left = location[0]
            tl = (left, top)
            br = (right, bottom)
            color = (255, 0, 0)
            thickness = 2
            img_save = cv2.rectangle(img_save, tl, br, color, thickness)
            cv2.putText(img_save, label, tl, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite('res_img.jpg', img_save)

    def run(self, path_img_test='./img_test/angry.jpg'):
        img = cv2.imread(path_img_test)
        face_locations = face_recognition.face_locations(img)
        list_face = []
        for face_location in face_locations:
            # Print the location of each face in this image
            top, right, bottom, left = face_location
            face_image = img[top:bottom, left:right]
            cv2.imwrite('face.jpg', face_image)
            face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)
            list_face.append(face_image)
        batch_img = np.array(list_face).reshape((-1, 48, 48, 1))
        labels_res = self.mapper[self.model_emotion_classification.predict_classes(batch_img)[0]]
        print([face_locations], [labels_res])
        dict_res = zip([face_locations], [labels_res])
        self.draw_result(img, dict_res)


def main():
    path_model_emotion_classification = './weight/model.h5'
    path_img_test = './img_test/neutral.jpeg'
    emotion_detection = DetectEmotion(path_model_emotion_classification=path_model_emotion_classification)
    emotion_detection.run(path_img_test)


if __name__ == '__main__':
    main()
