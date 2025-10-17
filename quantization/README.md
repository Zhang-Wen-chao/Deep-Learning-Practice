# Neural-Network-Quantization
## [Introduction to Quantization on PyTorch](https://pytorch.org/blog/introduction-to-quantization-on-pytorch/)

### [Dynamic Quantization on BERT](https://pytorch.org/tutorials/intermediate/dynamic_quantization_bert_tutorial.html)
[这个博客](https://blog.csdn.net/zimiao552147572/article/details/105910915)写得挺详细。

基本run成功了。保存序列化文件遇到了问题，发邮件给有同样问题的人了。
### [Dynamic Quantization on an LSTM Word Language Model](https://pytorch.org/tutorials/advanced/dynamic_quantization_tutorial.html)
在colab都运行不了。

wikitext-2数据集也不知道去哪里下载带有validation的txt格式。
### [QUANTIZED TRANSFER LEARNING FOR COMPUTER VISION TUTORIAL](https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html)
这个在colab可以非常顺畅地运行。
### [FX GRAPH MODE POST TRAINING STATIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_static.html)
NameError: name 'qconfig_opt' is not defined

运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。

但imagenet的文件夹完整了。
Neural-Network-Quantization/FX_GRAPH_MODE_POST_TRAINING_STATIC_QUANTIZATION/data/imagenet
### [FX GRAPH MODE POST TRAINING DYNAMIC QUANTIZATION](https://pytorch.org/tutorials/prototype/fx_graph_mode_ptq_dynamic.html)
run成功了。
## [PYTORCH 2 EXPORT POST TRAINING QUANTIZATION](https://pytorch.org/tutorials/prototype/pt2e_quant_ptq.html)
NameError: name 'qconfig_opt' is not defined

运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。
### [PYTORCH 2 EXPORT QUANTIZATION-AWARE TRAINING (QAT)](https://pytorch.org/tutorials/prototype/pt2e_quant_qat.html)
NameError: name 'qconfig_opt' is not defined

运行不了：本教程描述了一个原型功能。 原型功能通常不作为 PyPI 或 Conda 等二进制发行版的一部分提供，除非有时在运行时标志后面，并且处于反馈和测试的早期阶段。
# 毕设进度
跟着追乐大兵的视频，看完了p13。这个系列的视频，重在分析量化方法。不在实现某个算法。

仔细看看怎么进行量化敏感层分析的。
## 王鑫宇仓库
### yolovp
(base) /dataset01/zwc/tensorrtx/yolop/build (master ✘)✹✭ ᐅ ./yolop -d yolop.trt /dataset01/zwc/tensorrtx/yolop/YOLOP/inference/images
140ms
### yolov5
qat
ptq 分两条路线： pytorch to onnx to engine; tensort api;

yolov5的qat，开始跑了，只有两个epoch，但还没导出onnx。现在确定导出的yolov5.onnx有qdq。
但是怎么没有校准呢？


非对称量化需要计算scale和偏移量。对称量化只需要计算scale，但是数据分布不饱和，是非饱和量化，好处是计算简单。
tensorrt，速度优先，ptq是优先选择速度快的，不管是int8还是fp16,是隐式量化，不可控制每一层。


我好像看明白腾讯云的那个文章里，是用yolov5自定义了一个calibrator，然后从onnx转到 trt int8。

export.py 好像没法用int8。
我好像看明白腾讯云的那个文章里，是用yolov5自定义了一个calibrator，然后从onnx转到 trt int8。

https://developer.aliyun.com/article/1258484

https://pytorch.org/TensorRT/_notebooks/vgg-qat.html


Yolo预处理：好像是会把图维持原尺寸比例   不拉伸的进行resize    
tensorrtx的预处理：就是把原图等比例缩放，贴到一块灰色背景图上，贴到中间位置。
mAPval values are for single-model single-scale on COCO val2017 dataset.
#### yolov5s
##### pytroch fp32 在 Coco val2017 上的 mAPval 
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml   
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=yolov5s.pt, batch_size=32, imgsz=640, conf_thres=0.001, iou_thres=0.6, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
val: New cache created: /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:38
                   all       5000      36335       0.67       0.52      0.565      0.371
Speed: 0.1ms pre-process, 1.0ms inference, 1.2ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp8/yolov5s_predictions.json...
loading annotations into memory...
Done (t=0.39s)
creating index...
index created!
Loading and preparing results...
DONE (t=4.81s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=53.68s).
Accumulating evaluation results...
DONE (t=11.77s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.571
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.402
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.625
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.723
Results saved to runs/val/exp8
##### pytroch fp32 在 Coco val2017 上的 mAPval 加上 iou conf
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=yolov5s.pt, batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:27
                   all       5000      36335      0.819      0.405      0.619      0.465
Speed: 0.1ms pre-process, 1.0ms inference, 0.7ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp9/yolov5s_predictions.json...
loading annotations into memory...
Done (t=0.59s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.75s).
Accumulating evaluation results...
DONE (t=1.66s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.329
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.390
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.441
Results saved to runs/val/exp9
##### pytroch fp16 在 Coco val2017 上的 mAPval 加上 iou conf
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=yolov5s.pt, batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:26
                   all       5000      36335      0.819      0.406      0.619      0.466
Speed: 0.1ms pre-process, 0.8ms inference, 0.8ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp10/yolov5s_predictions.json...
loading annotations into memory...
Done (t=0.62s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.62s).
Accumulating evaluation results...
DONE (t=1.66s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.329
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.307
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.442
Results saved to runs/val/exp10
batchsize = 32
fps = 1000/ (0.1+0.8+0.8) = 588.235
##### trt api fp16
在4090的卡上，batchsize = 8 是 fps 最大的，其次是16, 32。
怎么选取训练神经网络时的Batch size? - Summer Clover的回答 - 知乎
https://www.zhihu.com/question/61607442/answer/1875700191
###### batchsize = 8
Average Preprocess Time per Image: 109us
Average Inference Time per Image: 237us
Average NMS Time per Image: 31us
FPS: 2652.52
Total Images: 5000
Total Time: 1891941us

(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹✭ ᐅ python eval.py
loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.93s).
Accumulating evaluation results...
DONE (t=1.61s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.099
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436
###### batchsize = 16
Average Preprocess Time per Image: 110us
Average Inference Time per Image: 259us
Average NMS Time per Image: 30us
FPS: 2506.27
Total Images: 5000
Total Time: 1999175us

(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹✭ ᐅ python eval.py                                            
loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.87s).
Accumulating evaluation results...
DONE (t=1.64s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.099
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436
###### batchsize = 32
Average Preprocess Time per Image: 118us
Average Inference Time per Image: 252us
Average NMS Time per Image: 29us
FPS: 2506.27
Total Images: 5000
Total Time: 1998898us

(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹✭ ᐅ python eval.py                                                             
loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.81s).
Accumulating evaluation results...
DONE (t=1.58s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.099
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.327
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.436
###### batchsize = 64
Average Preprocess Time per Image: 135us
Average Inference Time per Image: 266us
Average NMS Time per Image: 29us
FPS: 2325.58
Total Images: 5000
Total Time: 2155015us

(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹✭ ᐅ python eval.py                                            
loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.12s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.79s).
Accumulating evaluation results...
DONE (t=1.59s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.382
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.099
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.326
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.386
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.224
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.106
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.435
##### yolov5s export fp16 engine
python export.py --weights yolov5s.pt --include engine --half --device 0 --iou-thres 0.45 --conf-thres 0.5                      

(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5s.engine --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5s.engine'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Loading yolov5s.engine for TensorRT inference...
[04/01/2024-03:50:11] [TRT] [I] Loaded engine size: 16 MiB
[04/01/2024-03:50:11] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +13, now: CPU 0, GPU 13 (MiB)
[04/01/2024-03:50:11] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +17, now: CPU 0, GPU 30 (MiB)
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/5000 00:00WARNING ⚠️ NMS time limit 0.550s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 00:59
                   all       5000      36335      0.818      0.406      0.619      0.466
Speed: 0.2ms pre-process, 0.6ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp11/yolov5s_predictions.json...
loading annotations into memory...
Done (t=0.56s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.23s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.49s).
Accumulating evaluation results...
DONE (t=1.65s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.277
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.386
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.109
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.331
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.226
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.308
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.434
Results saved to runs/val/exp11
##### trt api int8
###### batchsize = 1 , 10k calib
Average Preprocess Time per Image: 102us
Average Inference Time per Image: 512us
Average NMS Time per Image: 15us
FPS: 1589.83
Total Images: 5000
Total Time: 3153972us

(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹✭ ᐅ python eval.py
loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.55s).
Accumulating evaluation results...
DONE (t=1.39s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.116
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.153
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.134
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.195
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.092
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.125
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.121
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.216
###### batchsize = 1 , 1k calib
Average Preprocess Time per Image: 102us
Average Inference Time per Image: 510us
Average NMS Time per Image: 13us
FPS: 1600
Total Images: 5000
Total Time: 3131457us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.31s).
Accumulating evaluation results...
DONE (t=1.37s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.093
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.125
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.108
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.085
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.168
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.075
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.099
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.099
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.003
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.185
##### pytorch qat fp32
这个还不太对啊，仔细看看readme，还要加一些代码。

python pytorch_yolov5_qat.py --weights yolov5s.pt --cfg models/yolov5s.yaml --data data/coco.yaml --img 640 --device 0

2 epochs completed in 1.196 hours.
Optimizer stripped from runs/train/exp5/weights/last.pt, 14.8MB
Optimizer stripped from runs/train/exp5/weights/best.pt, 14.8MB

Validating runs/train/exp5/weights/best.pt...
Evaluating pycocotools mAP... saving runs/train/exp5/_predictions.json...
loading annotations into memory...
Done (t=0.46s)
creating index...
index created!
Loading and preparing results...
DONE (t=5.97s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=57.04s).
Accumulating evaluation results...
DONE (t=13.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.289
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.548
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.607
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.689
Results saved to runs/train/exp5
##### pytorch qat fp16 engine
python export.py --weights runs/train/exp5/weights/best.pt --include engine --half --device 0 --iou-thres 0.45 --conf-thres 0.5

Speed: 0.2ms pre-process, 0.7ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp21/best_predictions.json...
loading annotations into memory...
Done (t=0.65s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.05s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.46s).
Accumulating evaluation results...
DONE (t=1.64s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.371
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.212
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.290
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.108
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.409
Results saved to runs/val/exp21
#### yolov5n
##### trt api int8
###### batchsize = 32 , 1k calib
Average Preprocess Time per Image: 119us
Average Inference Time per Image: 115us
Average NMS Time per Image: 6us
FPS: 4166.67
Total Images: 5000
Total Time: 1210665us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=5.73s).
Accumulating evaluation results...
DONE (t=1.30s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.010
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.008
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.002
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.008
###### batchsize = 32 , 10k calib
Average Preprocess Time per Image: 126us
Average Inference Time per Image: 127us
Average NMS Time per Image: 6us
FPS: 3861
Total Images: 5000
Total Time: 1297712us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.02s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.00s).
Accumulating evaluation results...
DONE (t=1.30s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.007
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.010
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.009
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.010
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.012
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.007
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.010
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.012
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 118us
Average Inference Time per Image: 127us
Average NMS Time per Image: 19us
FPS: 3787.88
Total Images: 5000
Total Time: 1327766us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.33s).
Accumulating evaluation results...
DONE (t=1.52s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.199
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.054
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.212
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.150
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.056
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.230
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.291
##### yolov5n export fp16 engine
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5n.engine --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5n.engine'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Loading yolov5n.engine for TensorRT inference...
[04/08/2024-13:10:24] [TRT] [I] Loaded engine size: 5 MiB
[04/08/2024-13:10:24] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +3, now: CPU 0, GPU 3 (MiB)
[04/08/2024-13:10:24] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +9, now: CPU 0, GPU 12 (MiB)
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/5000 00:00WARNING ⚠️ NMS time limit 0.550s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 01:00
                   all       5000      36335      0.819      0.269       0.55      0.398
Speed: 0.2ms pre-process, 0.7ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp15/yolov5n_predictions.json...
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.03s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.42s).
Accumulating evaluation results...
DONE (t=1.49s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.216
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.199
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.058
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.235
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.291
Results saved to runs/val/exp15
##### yolov5n pytorch fp16
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5n.pt --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5n.pt'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5n summary: 213 layers, 1867405 parameters, 0 gradients, 4.5 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:23
                   all       5000      36335      0.816      0.268      0.547      0.394
Speed: 0.0ms pre-process, 0.6ms inference, 0.8ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp14/yolov5n_predictions.json...
loading annotations into memory...
Done (t=0.55s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.21s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.62s).
Accumulating evaluation results...
DONE (t=1.50s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.178
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.258
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.203
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.056
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.151
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.058
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.232
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.289
Results saved to runs/val/exp14
#### yolov5m
##### pytorch qat fp32
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.611
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.255
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.469
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.558
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.614
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.431
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.757
Results saved to runs/train/exp6
##### yolov5n pytorch fp16
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5m.pt --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5m.pt'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients, 48.9 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:30
                   all       5000      36335      0.815      0.497      0.667      0.526
Speed: 0.0ms pre-process, 1.2ms inference, 0.8ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp16/yolov5m_predictions.json...
loading annotations into memory...
Done (t=0.56s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.24s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.18s).
Accumulating evaluation results...
DONE (t=1.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.473
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.396
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.391
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.181
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.552
Results saved to runs/val/exp16
##### yolov5n export fp16 engine
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5m.engine --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half 
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5m.engine'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Loading yolov5m.engine for TensorRT inference...
[04/08/2024-14:08:47] [TRT] [I] Loaded engine size: 43 MiB
[04/08/2024-14:08:47] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +40, now: CPU 0, GPU 40 (MiB)
[04/08/2024-14:08:47] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +25, now: CPU 0, GPU 65 (MiB)
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/5000 00:00WARNING ⚠️ NMS time limit 0.550s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 01:05
                   all       5000      36335      0.815      0.495      0.666      0.523
Speed: 0.2ms pre-process, 0.9ms inference, 1.0ms NMS per image at shape (1, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp17/yolov5m_predictions.json...
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.21s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.57s).
Accumulating evaluation results...
DONE (t=1.71s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.394
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.168
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.487
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.278
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.389
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.547
Results saved to runs/val/exp17
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 123us
Average Inference Time per Image: 464us
Average NMS Time per Image: 38us
FPS: 1600
Total Images: 5000
Total Time: 3131118us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.49s).
Accumulating evaluation results...
DONE (t=1.73s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.345
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.150
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.404
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.379
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.381
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.161
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.441
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553 
##### trt api int8
###### batchsize = 32 , 1k calib
Average Preprocess Time per Image: 130us
Average Inference Time per Image: 297us
Average NMS Time per Image: 14us
FPS: 2267.57
Total Images: 5000
Total Time: 2209710us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.05s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.23s).
Accumulating evaluation results...
DONE (t=1.46s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.143
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.165
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.113
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.118
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.153
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.153
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.004
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.115
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.294
#### yolov5l
##### yolov5l pytorch fp16
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5l.pt --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5l.pt'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5l summary: 367 layers, 46533693 parameters, 0 gradients, 109.0 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:34
                   all       5000      36335      0.811      0.551      0.694      0.558
Speed: 0.0ms pre-process, 1.7ms inference, 0.9ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp19/yolov5l_predictions.json...
loading annotations into memory...
Done (t=0.55s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.25s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.25s).
Accumulating evaluation results...
DONE (t=1.77s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.204
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.557
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.313
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.623
Results saved to runs/val/exp19
##### yolov5l export fp16 engine
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5l.engine --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5l.engine'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Loading yolov5l.engine for TensorRT inference...
[04/08/2024-14:51:20] [TRT] [I] Loaded engine size: 91 MiB
[04/08/2024-14:51:20] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +88, now: CPU 0, GPU 88 (MiB)
[04/08/2024-14:51:20] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +33, now: CPU 0, GPU 121 (MiB)
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/5000 00:00WARNING ⚠️ NMS time limit 0.550s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 01:05
                   all       5000      36335      0.803      0.548       0.69      0.555
Speed: 0.2ms pre-process, 1.4ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp18/yolov5l_predictions.json...
loading annotations into memory...
Done (t=0.55s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.24s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.47s).
Accumulating evaluation results...
DONE (t=1.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.554
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.439
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.503
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.619
Results saved to runs/val/exp18
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 148us
Average Inference Time per Image: 781us
Average NMS Time per Image: 43us
FPS: 1028.81
Total Images: 5000
Total Time: 4866465us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.91s).
Accumulating evaluation results...
DONE (t=1.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.510
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.430
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.189
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.449
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.551
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.427
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.490
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.611
##### trt api int8
###### batchsize = 32, 1k calib
Average Preprocess Time per Image: 113us
Average Inference Time per Image: 411us
Average NMS Time per Image: 18us
FPS: 1845.02
Total Images: 5000
Total Time: 2720038us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.28s).
Accumulating evaluation results...
DONE (t=1.52s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.270
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.016
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.196
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.015
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.257
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.504
###### batchsize = 64, 1k calib
Average Preprocess Time per Image: 125us
Average Inference Time per Image: 408us
Average NMS Time per Image: 18us
FPS: 1814.88
Total Images: 5000
Total Time: 2763806us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.36s).
Accumulating evaluation results...
DONE (t=1.53s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.235
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.016
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.463
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.252
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.015
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.258
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.506
#### yolov5x
##### trt api int8
###### batchsize = 32
Average Preprocess Time per Image: 119us
Average Inference Time per Image: 697us
Average NMS Time per Image: 17us
FPS: 1200.48
Total Images: 5000
Total Time: 4169996us

(base) /dataset01/zwc/tensorrtx/yolov5/tensorrtx/yolov5/build (yolov5-v7.0 ✘)✹✭ ᐅ python eval.py                                                                                                                   
loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.60s).
Accumulating evaluation results...
DONE (t=1.48s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.176
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.226
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.164
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.323
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.137
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.188
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.189
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.170
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.351
##### trt api fp16
###### batchsize = 8
Average Preprocess Time per Image: 105us
Average Inference Time per Image: 1306us
Average NMS Time per Image: 47us
FPS: 685.871
Total Images: 5000
Total Time: 7297139us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.71s).
Accumulating evaluation results...
DONE (t=1.78s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
###### batchsize = 16
Average Preprocess Time per Image: 106us
Average Inference Time per Image: 1304us
Average NMS Time per Image: 46us
FPS: 686.813
Total Images: 5000
Total Time: 7285563us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.69s).
Accumulating evaluation results...
DONE (t=1.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
###### batchsize = 32
Average Preprocess Time per Image: 111us
Average Inference Time per Image: 1295us
Average NMS Time per Image: 45us
FPS: 689.18
Total Images: 5000
Total Time: 7262396us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.64s).
Accumulating evaluation results...
DONE (t=1.73s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.575
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.643
###### batchsize = 64
Average Preprocess Time per Image: 131us
Average Inference Time per Image: 1282us
Average NMS Time per Image: 45us
FPS: 685.871
Total Images: 5000
Total Time: 7293061us

loading annotations into memory...
Done (t=0.42s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.69s).
Accumulating evaluation results...
DONE (t=1.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.533
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.451
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.471
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.576
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.320
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.452
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.516
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.644
##### yolov5x export fp16 engine
python export.py --weights yolov5x.pt --include engine --half --device 0 --iou-thres 0.45 --conf-thres 0.5
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5x.engine --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5x.engine'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Loading yolov5x.engine for TensorRT inference...
[04/06/2024-11:59:42] [TRT] [I] Loaded engine size: 168 MiB
[04/06/2024-11:59:43] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in engine deserialization: CPU +0, GPU +165, now: CPU 0, GPU 165 (MiB)
[04/06/2024-11:59:43] [TRT] [I] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +1, GPU +41, now: CPU 1, GPU 206 (MiB)
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/5000 00:00WARNING ⚠️ NMS time limit 0.550s exceeded
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 5000/5000 01:09
                   all       5000      36335      0.802      0.579      0.705      0.568
Speed: 0.2ms pre-process, 2.1ms inference, 0.9ms NMS per image at shape (1, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp12/yolov5x_predictions.json...
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.21s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.29s).
Accumulating evaluation results...
DONE (t=1.82s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.420
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.545
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.228
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.578
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.469
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.245
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.648
Results saved to runs/val/exp12
##### yolov5x pytorch fp16
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ python val.py --save-json --weights yolov5x.pt --data /dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml --conf-thres 0.5 --iou-thres 0.45 --half
val: data=/dataset01/zwc/tensorrtx/yolov5/yolov5/data/coco.yaml, weights=['yolov5x.pt'], batch_size=32, imgsz=640, conf_thres=0.5, iou_thres=0.45, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=True, dnn=False
WARNING ⚠️ confidence threshold 0.5 > 0.001 produces invalid results
YOLOv5 🚀 v7.0-0-g915bbf29 Python-3.9.13 torch-2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217MiB)

Fusing layers... 
YOLOv5x summary: 444 layers, 86705005 parameters, 0 gradients, 205.5 GFLOPs
val: Scanning /dataset01/zwc/tensorrtx/yolov5/datasets/coco/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 157/157 00:38
                   all       5000      36335      0.806      0.579      0.707      0.572
Speed: 0.0ms pre-process, 2.7ms inference, 1.0ms NMS per image at shape (32, 3, 640, 640)

Evaluating pycocotools mAP... saving runs/val/exp13/yolov5x_predictions.json...
loading annotations into memory...
Done (t=0.51s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.21s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.99s).
Accumulating evaluation results...
DONE (t=1.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.422
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.230
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.481
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.330
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.466
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.248
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.530
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.653
Results saved to runs/val/exp13
### yolov6
wang-xinyu 没有 yolov6，追乐有。

还是暂时先放下这部分。
### yolov7
遇到问题
https://github.com/wang-xinyu/tensorrtx/issues/1405
我好像解决了，一会看看pytorch的结果，就知道了。
#### yolov7-tiny
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7-tiny.pt --save-json --name yolov7-tiny_640_val

Namespace(weights=['yolov7-tiny.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7-tiny_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 200 layers, 6219709 parameters, 6219709 gradients, 13.7 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

val: Scanning 'coco/val2017' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:02<00:00, 2271.34it/s]
val: New cache created: coco/val2017.cache
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:25<00:00,  6.09it/s]
                 all        5000       36335       0.802        0.39       0.373       0.276
Speed: 0.8/0.7/1.6 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7-tiny_640_val4/yolov7-tiny_predictions.json...
loading annotations into memory...
Done (t=0.54s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.24s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.46s).
Accumulating evaluation results...
DONE (t=1.63s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.276
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.228
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.343
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475
Results saved to runs/test/yolov7-tiny_640_val4
##### trt api fp16
###### batchsize = 16
Average Preprocess Time per Image: 117us
Average Inference Time per Image: 232us
Average NMS Time per Image: 27us
FPS: 2659.57
Total Images: 5000
Total Time: 1890453us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.48s).
Accumulating evaluation results...
DONE (t=1.68s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.284
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.324
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.432
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.359
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
###### batchsize = 32
Average Preprocess Time per Image: 131us
Average Inference Time per Image: 236us
Average NMS Time per Image: 27us
FPS: 2538.07
Total Images: 5000
Total Time: 1978237us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.60s).
Accumulating evaluation results...
DONE (t=1.69s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.284
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.314
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.094
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.323
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.433
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.317
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.100
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.358
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.491
#### yolov7
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✹✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7.pt --save-json --name yolov7_640_val         

Namespace(weights=['yolov7.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
RepConv.fuse_repvgg_block
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 306 layers, 36905341 parameters, 36905341 gradients, 104.5 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

the value of half: True
val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:48<00:00,  3.22it/s]
                 all        5000       36335       0.817       0.566        0.54       0.417
Speed: 2.3/1.2/3.4 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7_640_val/yolov7_predictions.json...
loading annotations into memory...
Done (t=0.56s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.52s).
Accumulating evaluation results...
DONE (t=1.79s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.416
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.327
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.661
Results saved to runs/test/yolov7_640_val
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 122us
Average Inference Time per Image: 859us
Average NMS Time per Image: 26us
FPS: 993.049
Total Images: 5000
Total Time: 5038487us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.01s).
Accumulating evaluation results...
DONE (t=1.78s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.403
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.524
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.446
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.188
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.587
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.443
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.444
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.201
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.502
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
##### trt api int8, 1k calib
###### batchsize = 32
Average Preprocess Time per Image: 127us
Average Inference Time per Image: 450us
Average NMS Time per Image: 9us
FPS: 1706.48
Total Images: 5000
Total Time: 2938152us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.05s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.91s).
Accumulating evaluation results...
DONE (t=1.46s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.253
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.222
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.157
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.399
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.167
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.005
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.163
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.430
#### yolov7x
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✹✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7x.pt --save-json --name yolov7x_640_val  

Namespace(weights=['yolov7x.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7x_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 354 layers, 71311741 parameters, 71311741 gradients, 189.7 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

the value of half: True
val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:45<00:00,  3.47it/s]
                 all        5000       36335       0.807       0.591       0.561       0.438
Speed: 3.2/1.0/4.2 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7x_640_val/yolov7x_predictions.json...
loading annotations into memory...
Done (t=0.55s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.26s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.73s).
Accumulating evaluation results...
DONE (t=1.85s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.437
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.559
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.479
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.237
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.487
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.344
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.481
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.254
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.538
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.691
Results saved to runs/test/yolov7x_640_val
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 123us
Average Inference Time per Image: 1303us
Average NMS Time per Image: 28us
FPS: 687.758
Total Images: 5000
Total Time: 7276385us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.16s).
Accumulating evaluation results...
DONE (t=1.80s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.423
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.547
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.621
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.222
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.683
##### trt api int8, 1k calib
###### batchsize = 32
Average Preprocess Time per Image: 142us
Average Inference Time per Image: 687us
Average NMS Time per Image: 14us
FPS: 1186.24
Total Images: 5000
Total Time: 4221411us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.55s).
Accumulating evaluation results...
DONE (t=1.58s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.267
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.348
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.309
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.289
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.498
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.223
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.288
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.029
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.304
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.546
#### yolov7-w6
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✹✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7-w6.pt --save-json --name yolov7_640_val

Namespace(weights=['yolov7-w6.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 343 layers, 70394300 parameters, 70394300 gradients, 89.9 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

the value of half: True
val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:36<00:00,  4.32it/s]
                 all        5000       36335       0.815       0.549       0.525        0.41
Speed: 1.8/0.9/2.7 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7_640_val2/yolov7-w6_predictions.json...
loading annotations into memory...
Done (t=0.52s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.25s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.39s).
Accumulating evaluation results...
DONE (t=1.77s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.409
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.523
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.456
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.325
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.447
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.182
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.497
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
Results saved to runs/test/yolov7_640_val2
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 132us
Average Inference Time per Image: 619us
Average NMS Time per Image: 42us
FPS: 1261.03
Total Images: 5000
Total Time: 3972222us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.97s).
Accumulating evaluation results...
DONE (t=1.80s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.441
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.155
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.450
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.322
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.438
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.698
##### trt api int8, 1k calib
###### batchsize = 32
Average Preprocess Time per Image: 127us
Average Inference Time per Image: 335us
Average NMS Time per Image: 19us
FPS: 2079
Total Images: 5000
Total Time: 2413286us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.38s).
Accumulating evaluation results...
DONE (t=1.53s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.247
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.314
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.279
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.012
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.208
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.011
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.225
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.560
#### yolov7-e6
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✹✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7-e6.pt --save-json --name yolov7-e6_640_val

Namespace(weights=['yolov7-e6.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7-e6_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 473 layers, 97202940 parameters, 97202940 gradients, 128.6 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

the value of half: True
val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [01:18<00:00,  2.01it/s]
                 all        5000       36335        0.82       0.576       0.549       0.429
Speed: 3.0/1.6/4.6 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7-e6_640_val/yolov7-e6_predictions.json...
loading annotations into memory...
Done (t=0.54s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.32s).
Accumulating evaluation results...
DONE (t=1.78s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.546
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.196
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.520
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.714
Results saved to runs/test/yolov7-e6_640_val
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 129us
Average Inference Time per Image: 976us
Average NMS Time per Image: 44us
FPS: 870.322
Total Images: 5000
Total Time: 5748225us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.06s).
Accumulating evaluation results...
DONE (t=1.80s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.413
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.455
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.331
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.453
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.507
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.706
##### trt api int8, 1k calib
###### batchsize = 32
Average Preprocess Time per Image: 127us
Average Inference Time per Image: 524us
Average NMS Time per Image: 22us
FPS: 1485.88
Total Images: 5000
Total Time: 3375584us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.55s).
Accumulating evaluation results...
DONE (t=1.57s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.354
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.312
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.048
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.543
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.231
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.293
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.294
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.268
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.586
#### yolov7-d6
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✹✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7-d6.pt --save-json --name yolov7-d6_640_val

Namespace(weights=['yolov7-d6.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7-d6_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 539 layers, 133757052 parameters, 133757052 gradients, 175.4 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

the value of half: True
val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:38<00:00,  4.04it/s]
                 all        5000       36335       0.816       0.581       0.553       0.432
Speed: 3.0/0.8/3.8 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7-d6_640_val/yolov7-d6_predictions.json...
loading annotations into memory...
Done (t=0.58s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.49s).
Accumulating evaluation results...
DONE (t=1.80s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.431
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.551
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.200
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.649
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.476
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.215
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.708
Results saved to runs/test/yolov7-d6_640_val
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 122us
Average Inference Time per Image: 1186us
Average NMS Time per Image: 45us
FPS: 739.098
Total Images: 5000
Total Time: 6775460us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.73s).
Accumulating evaluation results...
DONE (t=1.80s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.418
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.538
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.468
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.646
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.332
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.458
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.511
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.707
##### trt api int8, 1k calib
###### batchsize = 32
Average Preprocess Time per Image: 129us
Average Inference Time per Image: 625us
Average NMS Time per Image: 21us
FPS: 1290.32
Total Images: 5000
Total Time: 3881252us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.58s).
Accumulating evaluation results...
DONE (t=1.57s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.334
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.056
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.220
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.055
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.247
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
#### yolov7-e6e
##### pytorch fp16
###### batchsize=32
(base) /dataset01/zwc/20240408/tensorrtx/yolov7/yolov7 (main ✘)✹✭ ᐅ python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.5 --iou 0.45 --device 0 --weights yolov7-e6e.pt --save-json --name yolov7-d6_640_val

Namespace(weights=['yolov7-e6e.pt'], data='data/coco.yaml', batch_size=32, img_size=640, conf_thres=0.5, iou_thres=0.45, task='val', device='0', single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project='runs/test', name='yolov7-d6_640_val', exist_ok=False, no_trace=False, v5_metric=False)
YOLOR 🚀 v0.1-128-ga207844 torch 2.1.2+cu121 CUDA:0 (NVIDIA GeForce RTX 4090, 24217.3125MB)

Fusing layers... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/functional.py:504: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at ../aten/src/ATen/native/TensorShape.cpp:3526.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
Model Summary: 792 layers, 151687420 parameters, 151687420 gradients, 210.5 GFLOPS
 Convert model to Traced-model... 
/home/student001/anaconda3/lib/python3.9/site-packages/torch/nn/modules/module.py:844: UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). If you indeed want the .grad field to be populated for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations. (Triggered internally at aten/src/ATen/core/TensorBody.h:489.)
  if param.grad is not None:
 traced_script_module saved! 
 model is traced! 

the value of half: True
val: Scanning 'coco/val2017.cache' images and labels... 4952 found, 48 missing, 0 empty, 0 corrupted: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [00:00<?, ?it/s]
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 157/157 [00:42<00:00,  3.70it/s]
                 all        5000       36335       0.819       0.591       0.563       0.441
Speed: 3.6/0.8/4.4 ms inference/NMS/total per 640x640 image at batch-size 32

Evaluating pycocotools mAP... saving runs/test/yolov7-d6_640_val2/yolov7-e6e_predictions.json...
loading annotations into memory...
Done (t=0.58s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.29s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.39s).
Accumulating evaluation results...
DONE (t=1.82s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.561
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.482
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.346
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.482
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.485
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.534
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.724
Results saved to runs/test/yolov7-d6_640_val2
##### trt api fp16
###### batchsize = 32
Average Preprocess Time per Image: 122us
Average Inference Time per Image: 1515us
Average NMS Time per Image: 45us
FPS: 594.53
Total Images: 5000
Total Time: 8418291us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.16s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=9.09s).
Accumulating evaluation results...
DONE (t=1.81s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.428
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.549
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.474
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.183
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.480
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.665
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.468
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.524
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.725
##### trt api int8, 1k calib
###### batchsize = 32
Average Preprocess Time per Image: 121us
Average Inference Time per Image: 794us
Average NMS Time per Image: 22us
FPS: 1067.24
Total Images: 5000
Total Time: 4689560us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.51s).
Accumulating evaluation results...
DONE (t=1.55s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.340
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.302
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.243
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.550
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.282
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.018
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.253
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.601
### yolov8
#### yolov8n
##### trt api int8
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 103us
Average Inference Time per Image: 553us
Average NMS Time per Image: 15us
FPS: 1490.31
Total Images: 5000
Total Time: 3363805us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.71s).
Accumulating evaluation results...
DONE (t=1.43s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.180
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.145
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.124
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.338
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 104us
Average Inference Time per Image: 544us
Average NMS Time per Image: 2us
FPS: 1538.46
Total Images: 5000
Total Time: 3257242us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.04s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=6.78s).
Accumulating evaluation results...
DONE (t=1.47s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.161
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.205
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.021
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.120
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.309
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.145
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.174
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.020
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.124
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.338
##### trt api fp16
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 119us
Average Inference Time per Image: 1102us
Average NMS Time per Image: 5us
FPS: 815.661
Total Images: 5000
Total Time: 6137690us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.83s).
Accumulating evaluation results...
DONE (t=1.59s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.065
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.276
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.429
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.286
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.287
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.067
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.302
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.476
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 114us
Average Inference Time per Image: 1101us
Average NMS Time per Image: 32us
FPS: 801.925
Total Images: 5000
Total Time: 6244091us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.61s).
Accumulating evaluation results...
DONE (t=1.58s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.260
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.338
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.064
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.276
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.428
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.221
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.285
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.286
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.067
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.301
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475
#### yolov8s
##### trt api fp16
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 107us
Average Inference Time per Image: 781us
Average NMS Time per Image: 5us
FPS: 1119.82
Total Images: 5000
Total Time: 4471721us

loading annotations into memory...
Done (t=0.46s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.61s).
Accumulating evaluation results...
DONE (t=1.80s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.443
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.134
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 107us
Average Inference Time per Image: 773us
Average NMS Time per Image: 39us
FPS: 1088.14
Total Images: 5000
Total Time: 4604459us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.14s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.41s).
Accumulating evaluation results...
DONE (t=1.69s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.379
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.384
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.535
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.283
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.376
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.134
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.418
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
#### yolov8m
##### trt api int8
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 105us
Average Inference Time per Image: 919us
Average NMS Time per Image: 2us
FPS: 974.659
Total Images: 5000
Total Time: 5142695us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.16s).
Accumulating evaluation results...
DONE (t=1.51s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.152
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.180
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.403
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 103us
Average Inference Time per Image: 920us
Average NMS Time per Image: 23us
FPS: 956.023
Total Images: 5000
Total Time: 5235313us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.06s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.25s).
Accumulating evaluation results...
DONE (t=1.50s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.213
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.234
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.152
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.377
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.180
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.160
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.403
##### trt api fp16
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 109us
Average Inference Time per Image: 1240us
Average NMS Time per Image: 6us
FPS: 738.007
Total Images: 5000
Total Time: 6784624us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.73s).
Accumulating evaluation results...
DONE (t=1.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.494
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.427
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.178
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 113us
Average Inference Time per Image: 1230us
Average NMS Time per Image: 45us
FPS: 720.461
Total Images: 5000
Total Time: 6951577us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.74s).
Accumulating evaluation results...
DONE (t=1.74s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.387
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.493
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.169
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.596
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.425
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.178
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.470
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.654
#### yolov8l
##### trt api fp16
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 109us
Average Inference Time per Image: 1728us
Average NMS Time per Image: 53us
FPS: 529.101
Total Images: 5000
Total Time: 9460296us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.74s).
Accumulating evaluation results...
DONE (t=1.77s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.448
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.450
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 109us
Average Inference Time per Image: 1710us
Average NMS Time per Image: 6us
FPS: 547.945
Total Images: 5000
Total Time: 9133796us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.77s).
Accumulating evaluation results...
DONE (t=1.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.412
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.521
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.453
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.186
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.633
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.449
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.451
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.198
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.496
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686
#### yolov8x
##### trt api int8
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 106us
Average Inference Time per Image: 1508us
Average NMS Time per Image: 38us
FPS: 605.327
Total Images: 5000
Total Time: 8266670us

loading annotations into memory...
Done (t=0.47s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.87s).
Accumulating evaluation results...
DONE (t=1.63s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.374
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 105us
Average Inference Time per Image: 1503us
Average NMS Time per Image: 4us
FPS: 620.347
Total Images: 5000
Total Time: 8066308us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.07s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=7.99s).
Accumulating evaluation results...
DONE (t=1.62s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.337
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.414
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.579
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.361
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.085
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.627
##### trt api fp16
###### batchsize = 1, cuda_post_process == "c"
Average Preprocess Time per Image: 115us
Average Inference Time per Image: 2446us
Average NMS Time per Image: 58us
FPS: 381.825
Total Images: 5000
Total Time: 13100031us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.86s).
Accumulating evaluation results...
DONE (t=1.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.696
###### batchsize = 1, cuda_post_process == "g"
Average Preprocess Time per Image: 114us
Average Inference Time per Image: 2443us
Average NMS Time per Image: 8us
FPS: 389.864
Total Images: 5000
Total Time: 12833491us

loading annotations into memory...
Done (t=0.43s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.15s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=8.76s).
Accumulating evaluation results...
DONE (t=1.76s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.531
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.194
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.459
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.461
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.206
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.697
### yolov9
#### yolov9-c
##### trt api int8
###### batchsize = 1
Average Preprocess Time per Image: 107us
Average Inference Time per Image: 1621us
Average NMS Time per Image: 70us
FPS: 556.174
Total Images: 5000
Total Time: 8997210us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.20s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=10.72s).
Accumulating evaluation results...
DONE (t=2.01s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.405
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.544
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.442
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.179
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.461
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.580
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.319
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.465
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.473
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.531
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.686
##### trt api fp16
###### batchsize = 1
Average Preprocess Time per Image: 113us
Average Inference Time per Image: 2159us
Average NMS Time per Image: 88us
FPS: 423.729
Total Images: 5000
Total Time: 11808605us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.31s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=12.07s).
Accumulating evaluation results...
DONE (t=2.18s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.622
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.507
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.519
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.354
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.527
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.539
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.310
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.719
#### yolov9-e
##### trt api int8
###### batchsize = 1
Average Preprocess Time per Image: 105us
Average Inference Time per Image: 1307us
Average NMS Time per Image: 56us
FPS: 681.199
Total Images: 5000
Total Time: 7348383us

loading annotations into memory...
Done (t=0.45s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.19s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=10.24s).
Accumulating evaluation results...
DONE (t=1.86s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.306
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.436
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.342
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.185
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.434
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.238
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.205
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.501
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.329
##### trt api fp16
###### batchsize = 1
Average Preprocess Time per Image: 106us
Average Inference Time per Image: 1646us
Average NMS Time per Image: 84us
FPS: 544.662
Total Images: 5000
Total Time: 9186770us

loading annotations into memory...
Done (t=0.44s)
creating index...
index created!
Loading and preparing results...
DONE (t=0.30s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=11.66s).
Accumulating evaluation results...
DONE (t=2.16s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.448
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.617
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.490
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.513
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.588
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.342
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.514
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.525
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.306
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.593
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
# 追乐大兵的b站视频
## 环境安装成功
```bash
docker pull nvcr.io/nvidia/tensorrt:22.12-py3
sudo docker run -it -d \
    --name mmyolo_trt8.5.1 \
    -v /dataset01:/dataset01 \
    --network="host" \
    --gpus all \
    nvcr.io/nvidia/tensorrt:22.12-py3

sudo docker exec -it mmyolo_trt8.5.1 /bin/bash

apt install ninja-build
apt-get install libgl1-mesa-glx
```
## p4 mmyolo初体验
### 训练
```bash
sudo docker exec -it mmyolo_trt8.5.1 /bin/zsh
cd /dataset01/zwc/mmyolo-hb/mmyolo
python tools/train.py configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.745
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.942
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.876
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.745
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.697
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.797
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.813
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813
03/06 14:13:12 - mmengine - INFO - bbox_mAP_copypaste: 0.745 0.942 0.876 -1.000 -1.000 0.745
03/06 14:13:12 - mmengine - INFO - Epoch(val) [40][28/28]    coco/bbox_mAP: 0.7450  coco/bbox_mAP_50: 0.9420  coco/bbox_mAP_75: 0.8760  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.7450  data_time: 0.0873  time: 0.1076
03/06 14:13:12 - mmengine - INFO - The previous best checkpoint /dataset01/zwc/mmyolo-hb/mmyolo/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/best_coco_bbox_mAP_epoch_30.pth is removed
03/06 14:13:15 - mmengine - INFO - The best checkpoint with 0.7450 coco/bbox_mAP at 40 epoch is saved to best_coco_bbox_mAP_epoch_40.pth.
### 导出onnx
```bash
python projects/easydeploy/tools/export.py \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --work-dir work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
    --opset 11 \
    --backend 1 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25
```
ONNX export success, save into work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx
### onnx用cpu推理
```bash
python projects/easydeploy/tools/image-demo.py \
    data/cat/images/IMG_20210728_205117.jpg \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx \
    --device cpu
# 推理结果在下面目录：
/dataset01/zwc/mmyolo-hb/mmyolo/output/IMG_20210728_205117.jpg
```
### 导出 tensorrt8 部署需要的onnx
```bash
python projects/easydeploy/tools/export.py \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
    --work-dir work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --img-size 640 640 \
    --batch 1 \
    --device cuda:0 \
    --simplify \
    --opset 11 \
    --backend 2 \
    --pre-topk 1000 \
    --keep-topk 100 \
    --iou-threshold 0.65 \
    --score-threshold 0.25
```
ONNX export success, save into work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx
### 用英伟达显卡生成序列化文件
```bash
python projects/easydeploy/tools/build_engine.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0
```
Save in /dataset01/zwc/mmyolo-hb/mmyolo/work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.engine
### 用英伟达显卡进行图片推理
```bash
python projects/easydeploy/tools/image-demo.py \
    data/cat/images/IMG_20210728_205312.jpg \
    configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
    work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/end2end.engine \
    --device cuda:0
# 推理结果在下面目录：
/dataset01/zwc/mmyolo-hb/mmyolo/output/IMG_20210728_205312.jpg
```
## 导出mmyolo系列模型,onnx tensorrt版本
### yolov5
就还是用mmyolo初体验中，猫检测器的转换命令
```bash
python projects/easydeploy/tools/export.py \
	configs/yolov5/yolov5_s-v61_fast_1xb12-40e_cat.py \
	work_dirs/yolov5_s-v61_fast_1xb12-40e_cat/epoch_40.pth \
	--work-dir work_dirs/yolov5_s-v61_fast_1xb12-40e_cat \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### yolov6
```bash
pip install openmim==0.3.7
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir
pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 30000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

netron work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/end2end.onnx
ssh -L 8080:localhost:8080 student001@10.20.30.160
# backend 2 是trt8, backend 3 是trt7;
# yolov6 用trt8, 检测头是EfficientNMS_TRT; 用 trt7 是BatchedNMSDynamic_TRT;
```
### yolov7
```bash
workdir="yolov7_tiny_syncbn_fast_8x16b-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### yolov8
```bash
workdir="yolov8_n_syncbn_fast_8xb16-500e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### yolovx
```bash
workdir="yolox_tiny_fast_8xb8-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 3 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### ppyoloe_plus
```bash
workdir="ppyoloe_plus_s_fast_8xb8-80e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### rtmdet
```bash
workdir="rtmdet_tiny_syncbn_fast_8xb32-300e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
### rtmdet 旋转框
```bash
workdir="rtmdet-r_tiny_fast_1xb8-36e_dota"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir


pth_path=`ls ./work_dirs/$workdir/*.pth`
python projects/easydeploy/tools/export.py \
	work_dirs/$workdir/$workdir.py \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 100 \
	--iou-threshold 0.65 \
	--score-threshold 0.25
```
这个导出有问题。
## 初步导出Tensorrt
```bash
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

python projects/easydeploy/tools/export.py \
	$config_file \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 1000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.3

python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0

python projects/easydeploy/tools/image-demo.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    --device cuda:0
```
## TensorRT模型精度的验证
```bash
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
mim download mmyolo --config $workdir --dest ./work_dirs/$workdir

pth_path=`ls ./work_dirs/$workdir/*.pth`

config_file="work_dirs/$workdir/$workdir.py"

python projects/easydeploy/tools/export.py \
	$config_file \
	${pth_path} \
	--work-dir work_dirs/$workdir \
    --img-size 640 640 \
    --batch 1 \
    --device cpu \
    --simplify \
	--opset 11 \
	--backend 2 \
	--pre-topk 30000 \
	--keep-topk 300 \
	--iou-threshold 0.65 \
	--score-threshold 0.001

python projects/easydeploy/tools/build_engine.py \
     ./work_dirs/$workdir/end2end.onnx \
    --img-size 640 640 \
    --device cuda:0

python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    demo/dog.jpg \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0

python self_study_scripts/01cmp_output_with_torch_tensorrt.py \
    /dataset01/coco2017/val2017/ \
    $config_file \
    ./work_dirs/$workdir/end2end.engine \
    $pth_path \
    --device cuda:0 \
	--result-json-path /dataset01/zwc/mmyolo-hb/mmyolo/self_study_scripts/tensort-1.json
这个命令需要很久，要推理5000个呢。
```
目标：
1. 探索相同输入下tensorrt和pytorch两个框架的输出有无不同
2. 针对 coco-val 数据集下的精度，tensorrt和pytorch得到的有何区别？

得到yolov6 pytorch版本输出

```bash
# 测试精度， 为加快速度还请去掉'/dataset01/zwc/mmyolo-hb/mmyolo/work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco.py'文件中的 visualization hook
# visualization=dict(type='mmdet.DetVisualizationHook')
workdir="yolov6_n_syncbn_fast_8xb32-400e_coco"
pth_path=`ls ./work_dirs/$workdir/*.pth`
python tools/test.py work_dirs/$workdir/$workdir.py \
                     $pth_path \
					 --json-prefix ./self_study_scripts/yolov6_n_coco_result

python ./self_study_scripts/02evel_trt_json_output.py
```
第一个是pytorch，第二个是tensorrt.
loading annotations into memory...
Done (t=0.50s)
creating index...
index created!
number of coco_classes: 80
Loading and preparing results...
DONE (t=12.20s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=106.48s).
Accumulating evaluation results...
DONE (t=26.75s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.362
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.392
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.166
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.401
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.522
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.340
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.773

Loading and preparing results...
DONE (t=5.28s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=107.03s).
Accumulating evaluation results...
DONE (t=26.48s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.361
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.516
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.389
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.167
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.399
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.519
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.311
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.521
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.573
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.338
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.642
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.771
## 精度没有百分之百对齐的原因
```bash
python demo/image_demo.py demo/dog.jpg ./work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco.py ./work_dirs/yolov6_n_syncbn_fast_8xb32-400e_coco/yolov6_n_syncbn_fast_8xb32-400e_coco_20221030_202726-d99b2e82.pth
```
出现了版本差异导致的问题，后面的操作都跟不上了。这种分析也并不必写在论文里面。
# tips
## jupyter测试代理
```python
import requests

proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}
try:
    response = requests.get("https://www.google.com", proxies=proxies)
    print("Network connection is working.")
except requests.exceptions.RequestException as e: 
    print("Network connection is not working.")

```
## pytorch-quantization 安装报错

```bash
(base) /dataset01/zwc/tensorrtx/yolov5/yolov5 (v7.0 ✘)✭ ᐅ pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
Looking in indexes: https://pypi.org/simple, https://pypi.ngc.nvidia.com, https://pypi.ngc.nvidia.com
Collecting pytorch-quantization
  Downloading pytorch-quantization-2.2.1.tar.gz (6.8 kB)
  Preparing metadata (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py egg_info did not run successfully.
  │ exit code: 1
  ╰─> [16 lines of output]
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-xqi_gwa0/pytorch-quantization_e2fc7b1f484a451ba2a5f231b9ce6994/setup.py", line 137, in <module>
          raise RuntimeError(open("ERROR.txt", "r").read())
      RuntimeError:
      ###########################################################################################
      The package you are trying to install is only a placeholder project on PyPI.org repository.
      This package is hosted on NVIDIA Python Package Index.
      
      This package can be installed as:
      ```
      $ pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com pytorch-quantization
      ```
      ###########################################################################################
      
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: metadata-generation-failed

× Encountered error while generating package metadata.
╰─> See above for output.

note: This is an issue with the package mentioned above, not pip.
hint: See above for details.

# 解决办法
pip install --no-cache-dir --index-url https://pypi.nvidia.com pytorch-quantization
pip install sphinx-glpi-theme
pip install prettytable
```