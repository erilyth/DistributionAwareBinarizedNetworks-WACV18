# Code for - Distribution-Aware Binarization of Neural Networks for Sketch Recognition (WACV 18)

Deep neural networks are highly effective at a range of computational tasks. However, they tend to be computationally expensive, especially in vision-related problems, and also have large memory requirements. One of the most effective methods to achieve significant improvements in computational/spatial efficiency is to binarize the weights and activations in a network. However, naive binarization results in accuracy drops when applied to networks for most tasks. In this work, we present a highly generalized, distribution-aware approach to binarizing deep networks that allows us to retain the advantages of a binarized network, while reducing accuracy drops. We also develop efficient implementations for our proposed approach across different architectures. We present a theoretical analysis of the technique to show the effective representational power of the resulting layers, and explore the forms of data they model best. Experiments on popular datasets show that our technique offers better accuracies than naive binarization, while retaining the same benefits that binarization provides - with respect to run-time compression, reduction of computational costs, and power consumption.

### Cite
If you use our paper or repo in your work, please cite the original paper as:
```
@article{Prabhu2018DistributedBin,
  author  = {Ameya Prabhu, Vishal Batchu, Sri Aurobindo Munagala, Rohit Gajawada, Anoop Namboodiri},
  title   = {Distribution-Aware Binarization of Neural Networks for Sketch Recognition},
  journal = {IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year    = {2018}
}
```

### Usage instructions
* Clone the repo
* Install PyTorch and other required dependencies
* Run using,
`bash clean.sh; CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset='<dataset_name>' --data_dir='<dataset_path>' --nClasses=<num_classes> --workers=8 --epochs=<train_epochs> --batch-size=<batch_size> --testbatchsize=<test_batch_size> --learningratescheduler='<learningrate_scheduler>' --decayinterval=50 --decaylevel=2 --optimType='<optimizer>' --nesterov --tenCrop --maxlr=<max_learning_rate> --minlr=<min_learning_rate> --weightDecay=0 --binaryWeight --binStart=<bin_start> --binEnd=<bin_end> --model_def='<model_name>' --inpsize=<input_size> --name='<experiment_name>'`
* Parameters:
  * dataset - The name of the dataset that would be used for the experiment. Allowed datasets - 'tuberlin', 'cifar10', 'imagenet12', 'cifar100' and 'sketchyrecognition'.
  * data_dir - The directory where the dataset is stored
  * nClasses - Number of output classes for the dataset
  * epochs - Number of training epochs
  * batch-size - Training batch size
  * testbatchsize - Testing batch size
  * learningratescheduler - Allowed schedulers - 'decayschedular'
  * decayinterval - The interval of learning rate decay for the decayschedular
  * decaylevel - The value by which the learning rate is decayed at every decay interval
  * nesterov - Toogle nesterov momentum
  * tenCrop - Toggle ten crop on data. Crops each image at 10 locations making 10 copies that would all be passed through the network.
  * maxlr - The maximum allowed learning rate
  * minlr - The minimum allowed learning rate
  * weightDecay - The amount by which all the weights of the network are decayed at each epoch
  * binaryWeight - Binarize the weights of the layers specified between binStart and binEnd (both inclusive)
  * binStart - The starting layer for binarization of weights if binaryWeight is toggled on
  * binEnd - The ending layer for binarization of weights if binaryWeight is toggled on
  * model_def - Network model to use. Allowed models - 'alexnet', 'alexnetfbin', 'alexnethybrid', 'alexnethybridv2', 'alexnetwbin', 'googlenet', 'googlenetfbin', 'googlenetwbin', 'mobilenet', 'nin', 'resnet18', 'resnetfbin18', 'resnethybrid18', 'resnethybridv218', 'resnethybridv318', 'resnetwbin18', n'sketchanet', 'sketchanetfbin', 'sketchanethybrid', 'sketchanethybridv2', 'sketchanetwbin', 'squeezenet', 'squeezenetfbin', 'squeezenethybrid', 'squeezenethybridv2', 'squeezenethybridv3', 'squeezenetwbin', 'vgg16_bncifar', 'densenet', 'bincifar' and 'bincifarfbin'.
  * inpsize - The input size of the image for the network (The network architecture would have to be adjusted accordingly). Defaults are 224 and 225 based on the network used.
  * name - The name of the experiment under which logs are saved.
* You could also take a look at the existing scripts (runx.sh) for samples
