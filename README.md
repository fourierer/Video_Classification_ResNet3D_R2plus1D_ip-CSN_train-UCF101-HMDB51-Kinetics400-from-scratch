# Video_Classification_ResNet3D_R2plus1D_ip-CSN_train-UCF101-HMDB51-Kinetics400-from-scratch
Using ResNet3D-50,R(2+1)D-50, and ip_CSN-50 to train UCD-101,HMDB-51 and Kinetics-400 from scratch.



此repo是 https://github.com/fourierer/Video_Classification_ResNet3D_Pytorch 的拓展，该repo不再使用在Kinetics上预训练好模型微调，而是使用ResNet3D，R(2+1)D，以及ip-CSN直接从头训练UCF-101，HMDB-51和Kinetics-400。

### 一、利用ResNet3D-50，R(2+1)D-50及ip-CSN-50（ir-CSN-50）从头训练UCF-101 

环境配置、数据预处理以及生成数据索引步骤与微调UCF-101部分一样，直接训练UCF-101即可。

1.使用ResNet3D-50训练

（1）训练命令

```shell
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --model resnet \
--model_depth 50 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --model resnet \
--model_depth 50 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型(和使用KInetics训练好的模型微调UCF一样的评测方法)

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --resume_path results/save_200.pth \
--model_depth 50 --n_classes 101 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data/ucf101_01.json /home/sunzheng/Video_Classification/data/results/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-1 accuracy
top-1 accuracy: 0.46418186624372193
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-3 accuracy
top-3 accuracy: 0.6330954269098599
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-5 accuracy
top-5 accuracy: 0.7079037800687286
```

可以看出ResNet3D在UCF-101从头开始训练，测试集中的top-1只有46.4%，但是训练集达到95%以上，严重过拟合。



2.使用R(2+1)D训练

（1）训练命令

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results_scratch_r2p1d --dataset ucf101 --model resnet2p1d \
--model_depth 50 --n_classes 101 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型(和使用Kinetics训练好的模型微调UCF一样的评测方法)

先运行：(指令要注明模型是resnet2p1d)

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model csnet
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results_scratch_r2p1d --dataset ucf101 --resume_path results_scratch_r2p1d/save_200.pth \
--model_depth 50 --n_classes 101 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model resnet2p1d
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data/ucf101_01.json /home/sunzheng/Video_Classification/data/results_scratch_r2p1d/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-1 accuracy
top-1 accuracy: 0.5659529473962464
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-3 accuracy
top-3 accuracy: 0.7287866772402855
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-5 accuracy
top-5 accuracy: 0.7919640496960084
```





3.使用ip-CSN-50训练

（1）训练命令

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --model csnet \
--model_depth 50 --n_classes 101 --batch_size 128 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型(和使用Kinetics训练好的模型微调UCF一样的评测方法)

先运行：(指令要注明模型是csnet)

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model csnet
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --resume_path results/save_200.pth \
--model_depth 50 --n_classes 101 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model csnet
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data/ucf101_01.json /home/sunzheng/Video_Classification/data/results/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-1 accuracy
top-1 accuracy: 0.556436690457309
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-3 accuracy
top-3 accuracy: 0.7224425059476606
```

```python
load ground truth
number of ground truth: 3783
load result
number of result: 3783
calculate top-5 accuracy
top-5 accuracy: 0.7827121332275971
```

可以看出ip-CSN-3D在UCF-101从头开始训练，测试集中的top-1只有55.64%，但是训练集达到95%以上，也是严重过拟合。





### 二、利用ResNet3D-50及ip-CSN-50（ir-CSN-50）从头训练HMDB-51

环境配置、数据预处理以及生成数据索引步骤与微调HMDB-51部分一样，直接训练HMDB-51即可。

1.使用ResNet3D-50训练

（1）训练命令

```shell
python main.py --root_path ~/data --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results --dataset hmdb51 --model resnet \
--model_depth 50 --n_classes 51 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_scratch_r3d --dataset hmdb51 --model resnet \
--model_depth 50 --n_classes 51 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型(和使用Kinetics训练好的模型微调HMDB-51一样的评测方法)

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_scratch_r3d --dataset hmdb51 --resume_path results_scratch_r3d/save_200.pth \
--model_depth 50 --n_classes 51 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data_hmdb/hmdb51_1.json /home/sunzheng/Video_Classification/data_hmdb/results_scratch_r3d/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-1 accuracy
top-1 accuracy: 0.2235294117647059
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-3 accuracy
top-3 accuracy: 0.40326797385620916
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-5 accuracy
top-5 accuracy: 0.5078431372549019
```



2.使用R(2+1)D训练

（1）训练命令

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_scratch_r2p1d --dataset hmdb51 --model resnet2p1d \
--model_depth 50 --n_classes 51 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型(和使用Kinetics训练好的模型微调HMDB一样的评测方法)

先运行：(指令要注明模型是resnet2p1d)

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model resnet2p1d
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_scratch_r2p1d --dataset hmdb51 --resume_path results_scratch_r2p1d/save_200.pth \
--model_depth 50 --n_classes 51 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model resnet2p1d
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data_hmdb/hmdb51_1.json /home/sunzheng/Video_Classification/data_hmdb/results_scratch_r2p1d/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-1 accuracy
top-1 accuracy: 0.2169934640522876
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-3 accuracy
top-3 accuracy: 0.42549019607843136
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-5 accuracy
top-5 accuracy: 0.5294117647058824
```



3.使用ip-CSN-50训练

（1）训练命令

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 50 --n_classes 700 --batch_size 128 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_scratch_csn --dataset hmdb51 --model csnet \
--model_depth 50 --n_classes 51 --batch_size 32 --n_threads 4 --checkpoint 5
```



（2）评测训练好的模型(和使用Kinetics训练好的模型微调UCF一样的评测方法)

先运行：(指令要注明模型是csnet)

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 700 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model csnet
```

例如在我的服务器上为：

```shell
python main.py --root_path /home/sunzheng/Video_Classification/data_hmdb --video_path hmdb51_videos/jpg --annotation_path hmdb51_1.json \
--result_path results_scratch_csn --dataset hmdb51 --resume_path results_scratch_csn/save_200.pth \
--model_depth 50 --n_classes 51 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1 --model csnet
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /home/sunzheng/Video_Classification/data_hmdb/hmdb51_1.json /home/sunzheng/Video_Classification/data_hmdb/results_scratch_csn/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-1 accuracy
top-1 accuracy: 0.2529411764705882
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-3 accuracy
top-3 accuracy: 0.4562091503267974
```

```python
load ground truth
number of ground truth: 1530
load result
number of result: 1530
calculate top-5 accuracy
top-5 accuracy: 0.5549019607843138
```



### 三、利用ResNet3D-50及ip-CSN-50（ir-CSN-50）从头训练Kinetics 

**注：kinetics-400的训练和repo https://github.com/fourierer/Video_Classification_ResNet3D_Pytorch 中打架数据集的训练流程非常相似：（1）数据集中的视频格式不统一，kinetics中大部分是.mp4格式，小部分是.mkv和.mp4.webm，在打架数据集训练中提到的抽帧脚本对于一个数据集只能抽一个格式的视频，比如UCF-101中只能对.avi格式视频抽帧，kinetics中只能对.mp4格式的视频抽帧；（2）都是已经划分好训练集和测试集，没有索引文件，需要自行整理。**

1.下载Kinetics数据集（需翻墙）：https://www.dropbox.com/s/wcs01mlqdgtq4gn/compress.tar.gz?dl=1 （可能已经失效），数据集是压缩包共132G，包括训练集和验证集。

2.数据预处理

（1）对数据集的视频格式转换，统一转换成.mp4格式的视频

（和打架数据集一样，kinetics-400中视频格式不一致，大概有三种格式的视频，.mp4,.mkv以及.mp4.webm，还是使用打架行为识别的抽帧脚本来做，首先需要将kinetics-400中的视频统一转换成.mp4格式（这里是由于kinetics中大部分都是.mp4格式的视频））。通过改变脚本中的路径分别对训练集和测试集进行视频格式的转换。

视频格式转换脚本change_style.py：

```python
# 此脚本用于将视频目录下的所有非.mp4或者非.avi格式的视频转换为.avi格式的视频，用于抽帧

import argparse
import subprocess
import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--source_path', default='/data1/sz/data_kinetics/kinetics-400/val_256', type=str, help='Directory path of original videos')
parser.add_argument(
    '--dst_path', default='/data1/sz/data_kinetics/kinetics-400-mp4/val_256', type=str, help='Directory path of .mp4 videos') 
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
```

执行指令：

```shell
python change_style.py
```



（2）对训练集和测试集抽帧

对测试集抽帧：

```shell
python generate_video_jpgs.py /data1/sz/data_kinetics/kinetics-400-mp4/val_256/ /data1/sz/data_kinetics/kinetics400_videos/jpg/val_256/ kinetics
```

对训练集抽帧：

```shell
python generate_video_jpgs.py /data1/sz/data_kinetics/kinetics-400-mp4/train_256/ /data1/sz/data_kinetics/kinetics400_videos/jpg/train_256/ kinetics
```



（3）利用.jpg文件生成数据集索引.json文件

和打架行为识别一样，三个步骤生成打架数据集索引文件，1）利用make_index.py脚本生成trainlist01.list和testlist01.txt；2）将划分好的打架数据集中的训练集和测试集放在一起；3）生成kinetics数据集的classInd.txt文件。

1）利用make_index.py脚本生成trainlist01.list和testlist01.txt，然后手动复制另外的trainlist02.txt和testlist02.txt，trainlist03.txt和testlist03.list。

make_index.py

```python
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

```

运行指令：

```shell
python make_index.py --is_train 1 # 生成trainlist01.txt
python make_index.py --is_train 0 # 生成testlist01.txt
```



2）将kinetics-400的训练集和测试集放到一起

由于kinetics数据集有400个类别，手动移动不太现实，这里采用move_train_test_together.py脚本来自动合并训练集和测试集：

move_train_test_together.py：

```python
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
```

直接运行：

```shell
python move_train_test_together.py
```



3）生成kinetics数据集的classInd.txt索引文件

make_classind.py

```python
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
```

运行指令：

```shell
python make_classind.py
```



以上完成以后，利用ucf101_json.py脚本生成.json文件：

```shell
python -m utils.ucf101_json /data1/sz/data_kinetics/kinetics-400-mp4 /data1/sz/data_kinetics/kinetics400_videos/jpg_mix_kinetics/ /data1/sz/data_kinetics/
```

**注：这里仍然采用ucf101的生成索引的方式，因为kinetics_json.py没有跑通。**



3.训练

**使用ResNet3D-50训练（由于kinetics-400训练时间太长，这里只做一个ResNet3D的尝试，四张RTX 2080Ti显卡，batch_size设置为128）**

（1）训练命令

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics400_1.json \
--result_path results --dataset ucf101 --model resnet \
--model_depth 50 --n_classes 400 --batch_size 32 --n_threads 4 --checkpoint 5
```

例如在我的服务器上为：

```shell
python main.py --root_path /data1/sz/data_kinetics --video_path kinetics400_videos/jpg_mix_kinetics --annotation_path ucf101_01.json \
--result_path results_scratch_r3d --dataset ucf101 --model resnet \
--model_depth 50 --n_classes 400 --batch_size 32 --n_threads 4 --checkpoint 5
```

**注：这里--dataset的参数仍然采用ucf101，但实际上是kinetics的数据集**

（2）评测训练好的模型(和使用Kinetics训练好的模型微调HMDB-51一样的评测方法)

先运行：

```shell
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_200.pth \
--model_depth 50 --n_classes 400 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```

例如在我的服务器上为：

```shell
python main.py --root_path /data1/sz/data_kinetics --video_path kinetics400_videos/jpg_mix_kinetics --annotation_path ucf101_1.json \
--result_path results_scratch_r3d --dataset ucf101 --resume_path results_scratch_r3d/save_200.pth \
--model_depth 50 --n_classes 400 --n_threads 4 --no_train --no_val --inference --output_topk 5 --inference_batch_size 1
```



再运行：（Evaluate top-1 video accuracy of a recognition result(~/data/results/val.json).）

```shell
python -m util_scripts.eval_accuracy ~/data/kinetics.json ~/data/results/val.json --subset val -k 1 --ignore
```

例如在我的服务器上为：

```shell
python -m util_scripts.eval_accuracy /data1/sz/data_kinetics/ucf101_1.json /data1/sz/data_kinetics/results_scratch_r3d/val.json -k 1 --ignore
```

k代表top-k的准确率，输出top-1，top-3，top-5结果：

```python

```

```python

```

```python

```

