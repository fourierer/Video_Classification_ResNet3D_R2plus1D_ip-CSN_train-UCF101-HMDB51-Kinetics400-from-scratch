# 此脚本用于生成kinetics数据集的classInd.txt文件，格式如下：
# 1 xxx
# 2 xxx
# 3 xxx
# ...
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--sor_path', default='/data1/sz/data_kinetics/kinetics-400-mp4/train_256', type=str)
parser.add_argument(
    '--dst_path', default='/data1/sz/data_kinetics/kinetics-400-mp4/classInd.txt', type=str)
args = parser.parse_args()


def main():
    txt_path = args.dst_path
    count = 0 # 类别索引
    with open(txt_path, 'w') as f:
        for video_class in os.listdir(args.sor_path):
            count = count + 1
            f.write(str(count))
            f.write(' ')
            f.write(video_class)
            f.write('\n')
    
if __name__ == '__main__':
    main()

