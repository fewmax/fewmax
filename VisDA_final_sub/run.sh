# 8 gpus for imagenet
# 2 gpus for cifar, speech_commands
# 1 gpu  for covtype, higgs

#############
# table 1,4 #
#############

python main_pretext.py '/csiNAS/visda' --dataset visda -a resnet50 --cos --warm --lincls --tb --resume true --method 'MaxUP'  --imix icutmix   --proj mlp --temp 0.2 --epochs 50 --trial 134  --multiprocessing-distributed --dist-url 'tcp://localhost:10033' --lr 0.03 -b 64 --qlen 65536 --class-ratio 0.1
#python main_pretext.py '/csiNAS/imagenet/ILSVRC/Data/CLS-LOC' --dataset imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix icutmix   --proj mlp --temp 0.2 --epochs 800 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10031' --lr 0.03 -b 256 --qlen 65535 --class-ratio 0.1

#python main_pretext.py 'data' --dataset imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --imix icutmix --proj mlp --temp 0.2 --epochs 800 --trial 0 --multiprocessing-distributed --dist-url 'tcp://localhost:10032' --lr 0.03 -b 512 --qlen 65536 --class-ratio 0.1
