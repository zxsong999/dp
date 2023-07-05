import imageio
import os    

file_ori = r"Data\pic_display\DPP_before"
file_cur = r"Data\pic_display\DPP_after"

path_list = os.listdir(file_ori)
images = [imageio.imread(file_ori + '/' + str(img_path)) for img_path in path_list]
imageio.mimsave("ori.gif", images, "gif", duration=600, loop=1)

path_list = os.listdir(file_cur)
images = [imageio.imread(file_cur + '/' + str(img_path)) for img_path in path_list]
imageio.mimsave("cur.gif", images, "gif", duration=600, loop=1)

