import os
import random
import shutil
import sys


def makedir(new_dir):
    if not os.path.exists(new_dir):
        print("make new dir:", new_dir)
        os.mkdir(new_dir)


if __name__ == "__main__":

    # 获取项目根目录
    ABSPATH = os.path.abspath(sys.argv[0])
    ABSPATH = os.path.dirname(ABSPATH)
    print(ABSPATH)

    random.seed(1)

    # 总数据集
    dataset_dir = os.path.join(ABSPATH, "RMB_data")

    # 分割数据集
    split_dir = os.path.join(ABSPATH, "rmb_split")
    if not os.path.exists(split_dir):
        makedir(split_dir)

    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    dirlist = [train_dir, valid_dir, test_dir]
    for i in range(3):
        makedir(dirlist[i])

    print("dataset dir path:{}".format(dataset_dir))
    print("split dir path:j{}".format(split_dir))
    print("train dir path:{}".format(train_dir))
    print("valid dir path:{}".format(valid_dir))
    print("test dir path:{}".format(test_dir))

    # 图片占比
    train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    # 遍历总数据集文件夹
    for root, dirs, files in os.walk(dataset_dir):
        print("root:", root)
        for sub_dir in dirs:
            imgs = os.listdir(os.path.join(root, sub_dir))
            # 获取所有jpg后缀的文件
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))

            # 打乱数据
            random.shuffle(imgs)

            # 计算图片数量
            img_count = len(imgs)
            print("图片数量:", img_count)

            # 计算训练集索引的结束位置:1-80
            train_point = int(img_count * train_pct)
            print("train point:", train_point)

            # 计算验证集索引的结束位置:81-90
            valid_point = int(img_count * (train_pct + valid_pct))
            print("valid point:", valid_point)
            print("test point:", img_count - valid_point - train_point)

            # 把数据划分到训练集、验证集、测试集的文件夹
            for i in range(img_count):
                print(i)
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)

                elif train_point < i and i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])
                shutil.copy(src_path, target_path)
            print(
                'Class:{}, train:{}, valid:{}, test:{}'.format(
                    sub_dir,
                    train_point,
                    valid_point -
                    train_point,
                    img_count -
                    valid_point))
