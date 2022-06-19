import os

from PIL import Image

for colour in ["Fluor", "Green", "Purple", "Red"]:
    path = "single_cell/{}/others".format(colour)
    img_list = os.listdir(path)
    for cell in img_list:
        crop_img = Image.open(path + "/" + cell)
        w = crop_img.width
        h = crop_img.height
        patch = Image.new("RGB", (w, h))
        cell = cell.split("-")
        full_img_name = cell[0] + "-" + cell[2] + ".png"
        full_img_path = "full_image/{}/{}".format(colour, full_img_name)
        full_img = Image.open(full_img_path)
        upper_left = (int(float(cell[3]) - (w / 2)), int(float(cell[4][:-4]) - (h / 2)))
        full_img.paste(patch, box=upper_left)
        full_img.save(full_img_path)
        print("Patched {}".format(full_img_path))

