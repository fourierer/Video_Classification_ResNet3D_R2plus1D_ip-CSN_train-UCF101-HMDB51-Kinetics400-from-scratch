# 此脚本用于将视频目录下的所有非.mp4或者非.avi格式的视频转换为.avi格式的视频，用于抽帧

import argparse
import subprocess
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source_path', default='/data1/sz/data_kinetics/kinetics-400/train_256', type=str, help='Directory path of original videos')
parser.add_argument(
    '--dst_path', default='/data1/sz/data_kinetics/kinetics-400-mp4/train_256', type=str, help='Directory path of .mp4 videos') 
args = parser.parse_args()


def main():
    class_all = os.listdir(args.source_path)
    for video_class in tqdm.tqdm(class_all):
        #print(video_class)
        # 在目标文件夹中创建视频类别文件夹
        if not os.path.exists(os.path.join(args.dst_path, video_class)):
            os.makedirs(os.path.join(args.dst_path, video_class))
        video_path = os.path.join(args.source_path, video_class)
        #print(video_path)
        video_all = os.listdir(video_path)
        for video in tqdm.tqdm(video_all):
            '''
            name = video.split('.')[0] # 获取视频名称
            cmd = 'ffmpeg -i ' + video_path + '/' + video + ' ' + args.dst_path + '/' + video_class + '/' + name + '.mp4'
            subprocess.call(cmd, shell=True)
            '''
            
            name = video.split('.')[0] # 获取视频名称
            suffix = video.split('.')[1] # 获取视频后缀
            length = len(video.split('.'))
            if (suffix != 'mp4') | (length>2):
                #print(suffix)
                # 后缀不是mp4格式的视频(包括后缀是.mkv以及.mp4.webm,.mp4.mkv多个后缀的视频)，使用ffmpeg转换成.mp4格式的视频
                cmd = 'ffmpeg -i ' + video_path + '/' + video + ' ' + args.dst_path + '/' + video_class + '/' + name + '.mp4'
                subprocess.call(cmd, shell=True)
            else:
                # 后缀是mp4的直接复制
                cmd = 'cp ' + video_path + '/' + video + ' ' + args.dst_path + '/' + video_class + '/'
                subprocess.call(cmd, shell=True)
            
if __name__ == '__main__':
    main()


