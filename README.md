# CNN LSTM 
Implementation of CNN LSTM with Resnet backend for Video Classification
![alt text](https://raw.githubusercontent.com/HHTseng/video-classification/master/fig/CRNN.png)

# Getting Started
## Prerequisites
* PyTorch (ver. 0.4+ required)
* FFmpeg, FFprobe
* Python 3


### Try on your own dataset 

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

## Train
Once you have created the dataset, start training ->
```
python main.py --use_cuda --gpu 0 --batch_size 8 --n_epochs 100 --num_workers 0  --annotation_path ./data/annotation/ucf101_01.json --video_path ./data/image_data/  --dataset ucf101 --sample_size 150 --lr_rate 1e-4 --n_classes <num_classes>
```

## Note 
* All the weights will be saved to the snapshots folder 
* To resume Training from any checkpoint, Use
```
--resume_path <path-to-model> 
```


## Tensorboard Visualisation(Training for 4 labels from UCF-101 Dataset)
![alt text](https://github.com/pranoyr/cnn-lstm/blob/master/images/Screenshot%202020-08-13%20at%205.54.36%20PM.png)


## Inference
```
python inference.py  --annotation_path ./data/annotation/ucf101_01.json  --dataset ucf101 --model cnnlstm --n_classes <num_classes> --resume_path <path-to-model.pth> 
```

## References
* https://github.com/kenshohara/video-classification-3d-cnn-pytorch
* https://github.com/HHTseng/video-classification

## License
This project is licensed under the MIT License 

