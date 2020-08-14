import argparse
import tqdm
import os


parser = argparse.ArgumentParser()
parser.add_argument(
        '--is_train', default=None, type=int, help='make train(or test)list01.txt')
parser.add_argument('--dst_path', default='/data1/sz/data_kinetics/kinetics-400-mp4/', type=str, help='path to generate txt file')
args = parser.parse_args()

def make_index_txt(class_path, txt_path, train):
    cate = os.listdir(class_path)
    if train == 1:
        i = 0 # trainlist01.txt类别数
        with open(txt_path, 'w') as f:
            for is_fight in cate:
                i = i+1
                video_list = os.listdir(class_path + is_fight)
                for video in video_list:
                    f.write(is_fight + '/' + video + '.mp4' + ' ' + str(i) + '\n')
    elif train == 0:
        with open(txt_path, 'w') as f:
            for is_fight in cate:
                video_list = os.listdir(class_path + is_fight)
                for video in video_list:
                    f.write(is_fight + '/' + video + '.mp4' + '\n')



if __name__ == '__main__':
    # 类别必须使用抽帧之后的视频个数作为索引文件，因为kinetics400是根据视频的url来下载的
    # 有些视频在下载的时候很容易出现全黑的情况，此时无法抽帧
    if args.is_train == 1:
        class_path = '/data1/sz/data_kinetics/kinetics400_videos/jpg/train_256/'
        txt_path = args.dst_path + 'trainlist01.txt'
    elif args.is_train == 0:
        class_path = '/data1/sz/data_kinetics/kinetics400_videos/jpg/val_256/'
        txt_path = args.dst_path + 'testlist01.txt'
    make_index_txt(class_path, txt_path, args.is_train)







