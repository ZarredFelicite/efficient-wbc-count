import csv
import os
import pathlib
import random
import re
import shutil

path = "T:/FYP_dataset/mixed"
full_list = os.listdir(path)
full_list = list(set([x[:-4] for x in full_list]))
random.shuffle(full_list)
full_len = len(full_list)
test_val_len = int(full_len * 0.05)

test_list = full_list[:test_val_len]
val_list = full_list[test_val_len : test_val_len * 2]
train_list = full_list[test_val_len * 2 :]
with open(path + "/test.csv", "w", newline="") as f:
    for img in test_list:
        writer = csv.writer(f)
        writer.writerow([img + ".png", img + ".txt"])
with open(path + "/val.csv", "w", newline="") as f:
    for img in val_list:
        writer = csv.writer(f)
        writer.writerow([img + ".png", img + ".txt"])
with open(path + "/train.csv", "w", newline="") as f:
    for img in train_list:
        writer = csv.writer(f)
        writer.writerow([img + ".png", img + ".txt"])


def clip_yolo_annotations(path):
    r = re.compile(".*txt")
    full_list = list(filter(r.match, os.listdir(path)))
    for label in full_list:
        with open(path + "/" + label, "r") as f:
            annotations = f.readlines()
            new_annotations = []
            for ann in annotations:
                ann = [float(x) for x in ann.split(" ")]
                ann[1:] = [max(min(x, 0.999), 0.001) for x in ann[1:]]
                if any(x < 0.001 for x in ann[1:3]):
                    print("less than 0.0001")
                if any(x > 0.999 for x in ann[1:3]):
                    print("greater than 0.999")
                new_annotations.append(ann)
        with open(path + "/" + label, "w") as f:
            for ann in new_annotations:
                f.write(
                    "{} {} {} {} {}\n".format(ann[0], ann[1], ann[2], ann[3], ann[4])
                )


# clip_yolo_annotations(path="T:/FYP_dataset/mixed")


# for colour in ["Fluor", "Green", "Purple", "Red"]:
#     for cell in ["Basophil", "Monocyte", "Lymphocyte", "Eosonophil", "Neutrophil"]:
#         path = "Data/WBC_20220215/single_cell/{}/{}".format(colour, cell)
#         full_list = os.listdir(path)
#         random.shuffle(full_list)
#         full_len = len(full_list)
#         test_val_len = int(full_len * 0.1)
#         test_list = full_list[:test_val_len]
#         val_list = full_list[test_val_len : test_val_len * 2]
#         train_list = full_list[test_val_len * 2 :]
#         new_path_train = pathlib.Path(
#             "Data/WBC_20220215/single_split/{}/train/{}".format(colour, cell)
#         )
#         new_path_val = pathlib.Path(
#             "Data/WBC_20220215/single_split/{}/val/{}".format(colour, cell)
#         )
#         new_path_test = pathlib.Path(
#             "Data/WBC_20220215/single_split/{}/test/{}".format(colour, cell)
#         )
#         new_path_train.mkdir(parents=True, exist_ok=True)
#         new_path_val.mkdir(parents=True, exist_ok=True)
#         new_path_test.mkdir(parents=True, exist_ok=True)
#         for img in train_list:
#             shutil.copyfile(path + "/" + img, new_path_train.as_posix() + "/" + img)
#         for img in val_list:
#             shutil.copyfile(path + "/" + img, new_path_val.as_posix() + "/" + img)
#         for img in test_list:
#             shutil.copyfile(path + "/" + img, new_path_test.as_posix() + "/" + img)

# for colour in ["Green", "Purple", "Red"]:
#     path = "Data/WBC_20220215/full_image/{}".format(colour)
#     full_list = [x[:-4] for x in os.listdir(path + "/labels")]
#     random.shuffle(full_list)
#     full_len = len(full_list)
#     test_val_len = int(full_len * 0.1)
#     test_list = full_list[:test_val_len]
#     val_list = full_list[test_val_len : test_val_len * 2]
#     train_list = full_list[test_val_len * 2 :]
#     new_path_train_img = pathlib.Path(
#         "Data/WBC_20220215/full_image/{}/train/images".format(colour)
#     )
#     new_path_train_txt = pathlib.Path(
#         "Data/WBC_20220215/full_image/{}/train/labels".format(colour)
#     )
#     new_path_val_img = pathlib.Path(
#         "Data/WBC_20220215/full_image/{}/val/images".format(colour)
#     )
#     new_path_val_txt = pathlib.Path(
#         "Data/WBC_20220215/full_image/{}/val/labels".format(colour)
#     )
#     new_path_test_img = pathlib.Path(
#         "Data/WBC_20220215/full_image/{}/test/images".format(colour)
#     )
#     new_path_test_txt = pathlib.Path(
#         "Data/WBC_20220215/full_image/{}/test/labels".format(colour)
#     )
#     new_path_train_img.mkdir(parents=True, exist_ok=True)
#     new_path_train_txt.mkdir(parents=True, exist_ok=True)
#     new_path_val_img.mkdir(parents=True, exist_ok=True)
#     new_path_val_txt.mkdir(parents=True, exist_ok=True)
#     new_path_test_img.mkdir(parents=True, exist_ok=True)
#     new_path_test_txt.mkdir(parents=True, exist_ok=True)
#     for img in train_list:
#         print("train: ", img)
#         shutil.move(
#             path + "/images/" + img + ".png",
#             new_path_train_img.as_posix() + "/" + img + ".png",
#         )
#         shutil.move(
#             path + "/labels/" + img + ".txt",
#             new_path_train_txt.as_posix() + "/" + img + ".txt",
#         )
#     for img in val_list:
#         print("val: ", img)
#         shutil.move(
#             path + "/images/" + img + ".png",
#             new_path_val_img.as_posix() + "/" + img + ".png",
#         )
#         shutil.move(
#             path + "/labels/" + img + ".txt",
#             new_path_val_txt.as_posix() + "/" + img + ".txt",
#         )
#     for img in test_list:
#         print("test: ", img)
#         shutil.move(
#             path + "/images/" + img + ".png",
#             new_path_test_img.as_posix() + "/" + img + ".png",
#         )
#         shutil.move(
#             path + "/labels/" + img + ".txt",
#             new_path_test_txt.as_posix() + "/" + img + ".txt",
#         )
