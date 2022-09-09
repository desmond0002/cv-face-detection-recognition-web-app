import os
import cv2
from imutils import paths, resize
from source.face_recognition import recognize_faces
from flask import flash

def make_list(classes_dir, classifier_model_path, label_encoder_path):

    #faces_list = []
    image_paths = list(paths.list_images(classes_dir))
    index = 0
    total = 0
    class_index = 0
    class_index_total = 0
    class_names = os.listdir(classes_dir)
    class_num = []
    class_num_total = []
    # classifier_model_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "20_30REC-100p.pickle"
    # label_encoder_path = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "my_models" + os.sep + "20_30LAB-100p.pickle"

    for (i, image_path) in enumerate(image_paths):
        total += 1

        name = image_path.split(os.sep)[-2]

        flash("[INFO] processing image {}/{}".format(i + 1, len(image_paths)) + " - " + image_path.split(os.sep)[-1])

        image = cv2.imread(image_path)
        image = resize(image, width=600)

        faces = recognize_faces(image, classifier_model_path, label_encoder_path)

        class_index_total += 1

        if len(list(paths.list_images(os.path.dirname(image_path)))) == class_index_total:
            class_num.append(class_index)
            class_num_total.append(class_index_total)
            class_index_total = 1
            class_index = 0


        for g in faces:
            if name in g.values():
                class_index += 1
                index += 1
                break

    for o in range(len(class_names)):
        flash("{}: {}/{}".format(class_names[o], class_num[o], class_num_total[o]))
    flash("accuracy: {}".format(index / total * 100))

#classes_dir = "/home" + os.sep + "andrew" + os.sep + "diploma" + os.sep + "flask" + os.sep + "all_classes" + os.sep + "20_classes_30pic" + os.sep + "val_dir"
#make_list(classes_dir)

