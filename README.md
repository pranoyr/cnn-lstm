# CNN LSTM Video Classification
Implementation of CNN LSTM in Pytorch for Video Classification

## Getting Started
### Prerequisites
* PyTorch (ver. 0.4+ required)
* FFmpeg, FFprobe
* Python 3


### Download the UCF101 Dataset
https://www.crcv.ucf.edu/research/data-sets/ucf101/


## Try on your own dataset 

```
mkdir data
mkdir data/video_data
```
Put your video dataset inside data/video_data
It should be in this form --

```
+ data 
    + video_data    
            - bowling
            - walking
            + running 
                    - running0.avi
                    - running.avi
                    - runnning1.avi
```

Generate Images from the Video dataset
```
./utils/generate_data.sh
```

### Train
Once you have created the dataset, start training ->
```
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes 4
```

With Attention
```
python main.py --model cnnlstm_attn --n_classes 4 --use_cuda --gpu 0 --batch_size 32 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4
```

Also specifiy 
--n_classes <num_classes>. num_classese = equals the number of labels in your dataset.


## License
This project is licensed under the MIT License 

