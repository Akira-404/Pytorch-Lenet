import os
import random
import shutil
import sys

'''
随机分割数据
'''


def makedir(new_dir):
    if not os.path.exists(new_dir):
        print("create new dir:", new_dir)
        os.mkdir(new_dir)


def split_dataset():

    # 获取项目根目录
    ABSPATH = os.path.abspath(sys.argv[-1])
    ABSPATH = os.path.dirname(ABSPATH)
    print("当前项目根目录:{}".format(ABSPATH))

    # 总数据集:必须要有
    dataset_dir = os.path.join(ABSPATH, "RMB_data")

    # 分割数据集
    split_dir = os.path.join(ABSPATH, "rmb_split")

    # 清理数据
    if os.path.exists(split_dir):
        print("删除原有文件")
        shutil.rmtree(split_dir, True)

    random.seed(1)

    train_dir = os.path.join(split_dir, "train")
    valid_dir = os.path.join(split_dir, "valid")
    test_dir = os.path.join(split_dir, "test")

    dirlist = [split_dir, train_dir, valid_dir, test_dir]
    for i in range(len(dirlist)):
        makedir(dirlist[i])

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

            # 计算训练集索引的结束位置:0-80
            train_point = int(img_count * train_pct)
            print("train point:", train_point)

            # 计算验证集索引的结束位置:80-90
            valid_point = int(img_count * (train_pct + valid_pct))
            print("valid point:", valid_point)
            print("test point:", img_count * test_pct)

            # 把数据划分到训练集、验证集、测试集的文件夹
            for i in range(img_count):
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


if __name__ == "__main__":
    split_dataset()
