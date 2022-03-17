
# Commands Used

## To train the net

```
python tools/train_net.py --config-file output/config.yaml --num-gpus 2 
```
**When you just want do evaluation, using this:**
```
python tools/train_net.py --config-file output/config.yaml --num-gpus 2 --eval-only
```
**Or you want train the net continuing from an unfinished training, just using:**
```
python tools/train_net.py --config-file output/config.yaml --num-gpus 2 --resume
```
If you want change the solver, using resume will cause some wrong. One solution is loading the last saved weight file, changing the solver and retraining the net.
## To inference on the input image
Just run the simple demo:
```
python demo/demo_my.py --input Data/test --output test
```
## To use script augmenting data offline

```
python tools/test/rotate_from_json.py
```
**Now, the script just can rotate the image with it's annotations**  
  
The function will be redesign to support more augmentation method.

## To use tensorboard visuallize training data

```
tensorboard --logdir= path of your output dir
```
## To visulize the inference results saved in json file
```
python tools/visualize_json_results.py --input output/inference/coco_instances_results.json --output eval --dataset coco_filament_val
``` 
## To visualize the annotation data

```
python tools/visualize_data.py --source annotation --config-file configs/CondInst/MS_R_50_1x.yaml --output-dir output path
```

There are something wrong when using dataloader to visualize data.