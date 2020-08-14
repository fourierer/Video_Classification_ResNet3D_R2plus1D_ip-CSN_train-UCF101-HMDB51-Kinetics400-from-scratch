# 此脚本用于将/data1/sz/data_kinetics/kinetics400_videos/jpg/train_256以及
# /data1/sz/data_kinetics/kinetics400_videos/jpg/val_256中的400个类别的视频
# 复制到/data1/sz/data_kinetics/kinetics400_videos/jpg_mix/kinetics中

import argparse
import subprocess
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--sor_path', default='/data1/sz/data_kinetics/kinetics400_videos/jpg', type=str)
parser.add_argument(
    '--dst_path', default='/data1/sz/data_kinetics/kinetics400_videos/jpg_mix_kinetics', type=str)
args = parser.parse_args()


def main():
    count = 0
    for video_class in os.listdir(os.path.join(args.sor_path, 'train_256')):
        print(video_class)
        count = count + 1
        print(count)
        if not os.path.exists(os.path.join(args.dst_path, video_class)):
            os.mkdir(os.path.join(args.dst_path, video_class))
        sor_train = os.path.join(args.sor_path, 'train_256', video_class)
        sor_val = os.path.join(args.sor_path, 'val_256', video_class)
        dst = os.path.join(args.dst_path, video_class)
        cmd_train = 'cp -a ' + sor_train + '/.' + ' ' + dst
        cmd_val = 'cp -a ' + sor_val + '/.' + ' ' + dst
        subprocess.call(cmd_train, shell=True)
        subprocess.call(cmd_val, shell=True)

if __name__ == '__main__':
    main()



