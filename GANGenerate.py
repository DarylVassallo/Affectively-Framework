import sys,os
import glob

sys.path.append(os.path.dirname(sys.path[0]))

from GA.repair import *
from GAN.dcgan import Generator
from torch.autograd import Variable

from stable_baselines3 import PPO

import keyboard
import shutil

def get_level(noise, to_string, name, size, width):
    # print("noise: " + str(noise))
    # print("to_string: " + str(to_string))
    # print("name: " + str(name))
    # print("size: " + str(size))
    # print("get_level 1")
    # size += 1
    # width = 16 * 999

    model_to_load = name
    batch_size = 1
    image_size = 32 * size
    ngf = 64
    nz = 32
    z_dims = 10  # number different titles

    # print("get_level 2")

    generator = Generator(nz, ngf, image_size, z_dims)

    # print("get_level 2 - 1")

    generator.load_state_dict(torch.load(model_to_load, map_location=lambda storage, loc: storage))

    # print("get_level 2 - 2")

    latent_vector = torch.FloatTensor(noise).view(batch_size, nz, 1, 1)

    # print("get_level 3")

    with torch.no_grad():
        levels = generator(Variable(latent_vector))
    im = levels.data.cpu().numpy()
    im = np.argmax(im, axis=1)
    im = little_level(im[0], size)

    # print("get_level 4")

    if to_string:
        # print("im[0:14, 0:width]")
        # print(str(im[0:14, 0:width]))
        # print(str(im))

        # print("get_level 5")

        return arr_to_str(im[0:14, 0:width])
        # return arr_to_str(im)
    else:

        # print("get_level 6")

        return im[0:14, 0:width]
        # return im
    
def get_random_long_level(values_1, values_2):
    # print("STARTSTARTSTARTSTARTSTARTSTARTSTART")
    # print("STARTSTARTSTARTSTARTSTARTSTARTSTART")
    # print("STARTSTARTSTARTSTARTSTARTSTARTSTART")
    # print("STARTSTARTSTARTSTARTSTARTSTARTSTART")
    # print("STARTSTARTSTARTSTARTSTARTSTARTSTART")
    lvs = []

    long_segment = get_level(values_1, False, './GAN/generator.pth', 1, 28)
    short_segment = get_level(values_2, False, './GAN/generator.pth', 1, 28)

    # print("long_segment: " + str(long_segment))
    # print("short_segment: " + str(short_segment))
    # print("long_segment.shape: " + str(long_segment.shape))
    # print("short_segment.shape: " + str(short_segment.shape))

    lvs.append(np.concatenate([long_segment, short_segment],axis=1))

    lv = np.concatenate(lvs, axis=-1)
    lv = addLine(lv)

    # print("lv:")
    # print(str(lv))
    
    # print("ENDENDENDENDENDENDENDENDENDENDENDEND")
    # print("ENDENDENDENDENDENDENDENDENDENDENDEND")
    # print("ENDENDENDENDENDENDENDENDENDENDENDEND")
    # print("ENDENDENDENDENDENDENDENDENDENDENDEND")
    # print("ENDENDENDENDENDENDENDENDENDENDENDEND")
    return lv

def repair():
    net_name = rootpath + "//CNet//dict.pkl"
    lv_name = rootpath + "//LevelGenerator//GAN//Destroyed//lv0.txt"
    result_path = rootpath + "//GA//result"

    score, level = GA(net_name, lv_name, result_path, isfigure=True, isrepair=True)

    return level

def generateNewLevel(values_1, values_2):
    # destroyed_folder = os.path.join(os.path.dirname(__file__), "LevelGenerator", "GAN", "Destroyed")
    # if os.path.exists(destroyed_folder):
    #     for file in glob.glob(os.path.join(destroyed_folder, "*")):
    #         try:
    #             os.remove(file)
    #         except Exception as e:
    #             print(f"Could not delete {file}: {e}")
    # else:
    #     os.makedirs(destroyed_folder)

    # result_folder = os.path.join(os.path.dirname(__file__), "GA", "result")
    # target_files = ["result.txt", "start.txt"]
    # if os.path.exists(result_folder):
    #     for file in target_files:
    #         file_path = os.path.join(result_folder, file)
    #         if os.path.exists(file_path):
    #             try:
    #                 os.remove(file_path)
    #             except Exception as e:
    #                 print(f"Could not delete {file}: {e}")
    # else:
    #     os.makedirs(result_folder)

    # lvs = []

    lv = get_random_long_level(values_1, values_2)

    # cnt = calculate_broken_pipes(lv)
    # lvs.append((cnt, lv))
    # cnt_sum = 0
    # lv_path = os.path.join(destroyed_folder, f'lv0.txt')
    # with open(lv_path, 'w') as f:
    #     f.write(arr_to_str(lvs[0][1]))
    # cnt_sum += lvs[0][0]
    # level = repair()

    return lv

if __name__ == '__main__':
    values = json.loads(sys.argv[1])
    print("values")
    print(str(values))

    level = generateNewLevel(values, 0)
    print("final level")
    print(str(level))