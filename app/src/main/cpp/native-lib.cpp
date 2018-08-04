#include <jni.h>
#include <string>
#include <algorithm>
#define PROTOBUF_USE_DLLS 1
#define CAFFE2_USE_LITE_PROTO 1
#include <caffe2/predictor/predictor.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/timer.h>

#include "caffe2/core/init.h"
#include  "caffe2/core/tensor.h"

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <android/log.h>
#include "classes.h"
#define IMG_H 227
#define IMG_W 227
#define IMG_C 3
#define MAX_DATA_SIZE IMG_H * IMG_W * IMG_C
#define alog(...) __android_log_print(ANDROID_LOG_ERROR, "F8DEMO", __VA_ARGS__);

static caffe2::NetDef _initNet, _predictNet;
static caffe2::Predictor *_predictor;
static char raw_data[MAX_DATA_SIZE];
static float input_data[MAX_DATA_SIZE];
static caffe2::Workspace ws;

// A function to load the NetDefs from protobufs.
void loadToNetDef(AAssetManager* mgr, caffe2::NetDef* net, const char *filename) {
    AAsset* asset = AAssetManager_open(mgr, filename, AASSET_MODE_BUFFER);
    assert(asset != nullptr);
    const void *data = AAsset_getBuffer(asset);
    assert(data != nullptr);
    off_t len = AAsset_getLength(asset);
    assert(len != 0);
    if (!net->ParseFromArray(data, len)) {
        alog("Couldn't parse net from data.\n");
    }
    AAsset_close(asset);
}

extern "C"
void
Java_facebook_f8demo_ClassifyCamera_initCaffe2(
        JNIEnv* env,
        jobject /* this */,
        jobject assetManager) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);
    alog("Attempting to load protobuf netdefs...");
    loadToNetDef(mgr, &_initNet,   "squeeze_init_net_6.pb");
    loadToNetDef(mgr, &_predictNet,"squeeze_pred_net_6.pb");
    alog("done.");
    alog("Instantiating predictor...");
    _predictor = new caffe2::Predictor(_initNet, _predictNet);
    alog("done.")
}

float avg_fps = 0.0;
float total_fps = 0.0;
int iters_fps = 10;

extern "C"
JNIEXPORT jstring JNICALL
Java_facebook_f8demo_ClassifyCamera_classificationFromCaffe2(
        JNIEnv *env,
        jobject /* this */,
        jint h, jint w, jbyteArray Y, jbyteArray U, jbyteArray V,
        jint rowStride, jint pixelStride,
        jboolean infer_HWC) {
    env->NewStringUTF("Hello");
    if (!_predictor) {
        return env->NewStringUTF("Loading...");
    }
    jsize Y_len = env->GetArrayLength(Y);
    jbyte * Y_data = env->GetByteArrayElements(Y, 0);
    assert(Y_len <= MAX_DATA_SIZE);
    jsize U_len = env->GetArrayLength(U);
    jbyte * U_data = env->GetByteArrayElements(U, 0);
    assert(U_len <= MAX_DATA_SIZE);
    jsize V_len = env->GetArrayLength(V);
    jbyte * V_data = env->GetByteArrayElements(V, 0);
    assert(V_len <= MAX_DATA_SIZE);


    auto h_offset = std::max(0, (h - IMG_H) / 2);
    auto w_offset = std::max(0, (w - IMG_W) / 2);

    auto iter_h = IMG_H;
    auto iter_w = IMG_W;
    if (h < IMG_H) {
        iter_h = h;
    }
    if (w < IMG_W) {
        iter_w = w;
    }

    for (auto i = 0; i < iter_h; ++i) {
        jbyte* Y_row = &Y_data[(h_offset + i) * w];
        jbyte* U_row = &U_data[(h_offset + i) / 2 * rowStride];
        jbyte* V_row = &V_data[(h_offset + i) / 2 * rowStride];
        for (auto j = 0; j < iter_w; ++j) {
            // Tested on Pixel and S7.
            char y = Y_row[w_offset + j];
            char u = U_row[pixelStride * ((w_offset+j)/pixelStride)];
            char v = V_row[pixelStride * ((w_offset+j)/pixelStride)];
            
            float b_mean = 0.406, b_std = 0.225;
            float g_mean = 0.456, g_std = 0.224;
            float r_mean = 0.485, r_std = 0.229;

            auto b_i = 2 * IMG_H * IMG_W + j * IMG_W + i;
            auto g_i = 1 * IMG_H * IMG_W + j * IMG_W + i;
            auto r_i = 0 * IMG_H * IMG_W + j * IMG_W + i;

            if (infer_HWC) {
                b_i = (j * IMG_W + i) * IMG_C + 2;
                g_i = (j * IMG_W + i) * IMG_C + 1;
                r_i = (j * IMG_W + i) * IMG_C + 0;
            }
/*
  R = Y + 1.402 (V-128)
  G = Y - 0.34414 (U-128) - 0.71414 (V-128)
  B = Y + 1.772 (U-V)
 */
            input_data[r_i] = (-r_mean + std::min(255.0, std::max(0., y + 1.402 * (v - 128))) / 255.0) / r_std;
            input_data[g_i] = (-g_mean + std::min(255.0, std::max(0., y - 0.34414 * (u - 128) - 0.71414 * (v - 128))) / 255.0) / g_std;
            input_data[b_i] = (-b_mean + std::min(255.0, std::max(0., y + 1.772 * (u - v))) / 255.0) / b_std;
            }
    }
    caffe2::TensorCPU input;
    if (infer_HWC) {
        input.Resize(std::vector<int>({IMG_H, IMG_W, IMG_C}));
    } else {
        input.Resize(std::vector<int>({1, IMG_C, IMG_H, IMG_W}));
    }
    memcpy(input.mutable_data<float>(), input_data, IMG_H * IMG_W * IMG_C * sizeof(float));
    caffe2::Predictor::TensorVector input_vec{&input};
    caffe2::Predictor::TensorVector output_vec;
    caffe2::Timer t;
    t.Start();
    _predictor->run(input_vec, &output_vec);
    float fps = 1000/t.MilliSeconds();
    total_fps += fps;
    avg_fps = total_fps / iters_fps;
    total_fps -= avg_fps;

    constexpr int k = 6;
    // Find the top-k results manually.
    auto output = output_vec[0];
    std::vector<float> preds(output->size());
    float max_val = std::numeric_limits<float>::min();
    for(auto i=0; i < output->size(); ++i) {
      float tmp = output->template data<float>()[i];
      preds[i] = tmp;
      max_val = tmp > max_val ? tmp : max_val;
    }
    float exp_sum = 0.0;
    for(auto &val : preds) {
      val -= max_val;
      val = std::exp(val);
      exp_sum += val;
    }
    for(auto &val : preds)  {
      val /= exp_sum + 1e-10;
    }
    std::vector<int> idxs(preds.size());
    std::iota(idxs.begin(), idxs.end(), 0);
    std::sort(idxs.begin(), idxs.end(), 
        [&preds](const int &idx1, const int &idx2) {
          return preds[idx1] > preds[idx2];
        }
    );

    std::ostringstream stringStream;
    stringStream << avg_fps << " FPS\n";

    for (auto j = 0; j < k; ++j) {
        int idx = idxs[j];
        stringStream << j << ": " << imagenet_classes[idx] << " - " << preds[idx] * 100 << "%\n";
    }
    return env->NewStringUTF(stringStream.str().c_str());
}
