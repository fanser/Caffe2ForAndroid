## Caffe2ForAndroid

This project is based on [AICamera](https://github.com/caffe2/AICamera), which is a demo app. The app now supports six types classification: people, cup, mobile phone, computer, table and chair, which are collected from web.

![demo](https://github.com/fanser/Caffe2ForAndroid/raw/master/demo.gif)

### Introduction
I train the classification network by finetuning the SqueezeNet1_1, and this stage is done by Pytorch. Then the pytorch model is transformed to caffemodel style.

### Build
You can follow the introduction describled in [AICamera](https://github.com/caffe2/AICamera) to build this app.

* Note: This app uses the updated caffe2 (pytorch) version, so some static libraries, like `libcaffe2.a` etc., are rebuild on my machine, and they are different from the original [AICamera](https://github.com/caffe2/AICamera). In this project, we provide the all corresponding libraries and the pretrained model to help you work successfully. It is noticeable that if you replace by own model, the app may don't work, which probably caused by the conflict of caffe2 version. In this situation, you can try rebuild the libraries like libcaffe2.a, which your models depend on, and update these together.

### Tests

| Device             | Network       |  FPS  |
| ------------------ | ------------- | ----- |
| Huawei Honor 8     | SqueezeNet    |  7.5 |

### TODO
* Segmentation
* detection

### License

Please see the LICENSE file in the root directory the source tree.
