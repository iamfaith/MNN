//
//  benchmark.cpp
//  MNN
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#if defined(_MSC_VER)
#include <Windows.h>
#undef min
#undef max
#else
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#endif

#include "core/Backend.hpp"
#include <MNN/Interpreter.hpp>
#include <MNN/MNNDefine.h>
#include <MNN/Tensor.hpp>
#include <MNN/AutoTime.hpp>
#include "revertMNNModel.hpp"

//////////////////////// std drawing
#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_THREAD_LOCALS
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#include "ocv/std_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ocv/stb_image_write.h"
//////////////////////// std drawing

#include "utils.h"

class Object
{

public:
    int label;
    float prob;
    float x, y, width, height;
    float x1, y1;

    // Object(int label, float prob, float x, float y, float width, float height)
};

// 自定义比较函数，按照prob从大到小排序
bool compare_by_prob(const Object &a, const Object &b)
{
    return a.prob > b.prob;
}

void read_image(const char *imagepath, unsigned char **pixeldata)
{
    int w;
    int h;
    int c;
    int desired_channels = 3;
    *pixeldata = stbi_load(imagepath, &w, &h, &c, desired_channels);

    std::cout << sizeof(pixeldata) << " " << w << " " << h << " " << c << std::endl;
    // memcpy(imgmat->data, pixeldata, 3 * h * w);
    // stbi_image_free(pixeldata);
}

void displayStats(const std::vector<float> &costs, const std::string &name = "default", int quant = 0)
{
    float max = 0, min = FLT_MAX, sum = 0, avg;
    for (auto v : costs)
    {
        max = fmax(max, v);
        min = fmin(min, v);
        sum += v;
        // printf("[ - ] cost：%f ms\n", v);
    }
    avg = costs.size() > 0 ? sum / costs.size() : 0;
    std::string model = name;
    if (quant == 1)
    {
        model = "quant-" + name;
    }
    printf("[ - ] %-24s    max = %8.3f ms  min = %8.3f ms  avg = %8.3f ms\n", model.c_str(), max, avg == 0 ? 0 : min, avg);
}

int main(int argc, char *argv[])
{

    std::cout << "Yolo v5 test" << std::endl;
    unsigned char *pixel = NULL;
    char *imagepath = "/home/faith/AI_baili_train/images/22.png";
    int originalWidth;
    int originalHeight;
    int originChannel;
    if (argc == 1)
        // read_image(imagepath, &pixel);

        pixel = stbi_load(imagepath, &originalWidth, &originalHeight, &originChannel, 3);
    else
    {
        imagepath = argv[1];
        read_image(argv[1], &pixel);
    }

    const char *model_file = "/home/faith/best5000-sim.mnn";
    auto revertor = std::unique_ptr<Revert>(new Revert(model_file));
    int sparseBlockOC = 1;
    revertor->initialize(0, sparseBlockOC, false, true);
    auto modelBuffer = revertor->getBuffer();
    const auto bufferSize = revertor->getBufferSize();
    auto net = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromBuffer(modelBuffer, bufferSize));
    revertor.reset();
    net->setSessionMode(MNN::Interpreter::Session_Release);
    MNN::ScheduleConfig config;
    int numberThread = 16;
    config.numThread = numberThread;
    int forward = 0; // MNN_FORWARD_CPU;
    config.type = static_cast<MNNForwardType>(forward);
    MNN::BackendConfig backendConfig;
    int precision = 0;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision;
    backendConfig.power = MNN::BackendConfig::Power_High;
    config.backendConfig = &backendConfig;

    std::vector<float> costs;
    MNN::Session *session = net->createSession(config);

    MNN::Tensor *input = net->getSessionInput(session, "images");

    net->releaseModel();

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // 384.000000, 768.000000 h w
    const int targetWidth = 768;
    const int targetHeight = 384;

    int ori_h = originalHeight;
    int ori_w = originalWidth;
    float gain = fmin(targetHeight / (float)ori_h, targetWidth / (float)ori_w);
    float padw = (targetWidth - ori_w * gain) / 2;
    float padh = (targetHeight - ori_h * gain) / 2;
    // printf("\n----%f %f, %f, %f, %f\n", gain, padh, padw, img_h, img_w);

    int resize_h = (int)round((float)ori_h * gain);
    int resize_w = (int)round((float)ori_w * gain);

    // printf("\n----%f %f, %f, %d, %d\n", gain, padh, padw, resize_h, resize_w);

    void *newmat = malloc(sizeof(unsigned char) * resize_w * resize_h * originChannel);
    // preprocess input image
    resize_bilinear_c3((const unsigned char *)pixel, ori_w, ori_h, ori_w * 3, (unsigned char *)newmat, resize_w, resize_h, resize_w * 3);
    void *clast = from_cvmat2mat((const unsigned char *)newmat, resize_w, resize_h, resize_w * 3);

    // for (int i = 0; i < 500; i++) {
    // printf("%d ", ((unsigned char *)pixel)[i]);
    // printf("%f ", ((float *)clast)[i]);

    // std::cout << ((unsigned short *)clast)[i] << " ";
    // }

    int top = (int)(round(padh - 0.1));
    int bottom = (int)(round(padh + 0.1));
    int left = (int)(round(padw - 0.1));
    int right = (int)(round(padw + 0.1));

    int outw = resize_w + left + right;
    int outh = resize_h + top + bottom;
    float *padding_m = (float *)malloc(4 * outw * outh * originChannel);
    copy_make_border(padding_m, outw, outh, clast, resize_w, resize_h, originChannel, top, bottom, left, right, 114.f);

    // for (int i = 0, count = 0; count < 50; i++) {
    //     // std::cout << ((float *)clast)[i] << " ";
    //     if (i > top * resize_w + left) {
    //         printf("+%f ", ((float *)padding_m)[i]);
    //         count ++;
    //     }
    //     // printf("%d ", ((unsigned char *)imgmat->data)[i]);
    // }
    printf("\n");
    free(newmat);
    free(clast);
    int inputSize = outw * outh * originChannel;

    MNN::Tensor givenTensor(input, MNN::Tensor::CAFFE);
    auto inputData = givenTensor.host<float>();
    const float norm_val = 1 / 255.f;
    for (int i = 0; i < inputSize; ++i)
    {
        float _pixel = *(padding_m + i) * norm_val;
        inputData[i] = static_cast<float>(_pixel);
        // std::cout << inputData[i] << " ";
    }
    std::cout << inputSize << " " << originChannel << " " << resize_h << std::endl;
    input->copyFromHostTensor(&givenTensor);
    ///////////////////////////////////////////////////////////////////////////////////////////////

    const MNN::Backend *inBackend = net->getBackend(session, input);

    // create random tensor
    // std::shared_ptr<MNN::Tensor> givenTensor(MNN::Tensor::createHostTensorFromDevice(input, false));

    auto outputTensor = net->getSessionOutput(session, NULL);
    // std::shared_ptr<MNN::Tensor> expectTensor(MNN::Tensor::createHostTensorFromDevice(outputTensor, false));

    int warmup = 0;
    int loop = 1;

    // Warming up...
    for (int i = 0; i < warmup; ++i)
    {
        void *host = input->map(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType());
        input->unmap(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType(), host);

        net->runSession(session);

        host = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType());
        outputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType(), host);
    }

    for (int round = 0; round < loop; round++)
    {
        MNN::Timer _t;
        void *host = input->map(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType());
        input->unmap(MNN::Tensor::MAP_TENSOR_WRITE, input->getDimensionType(), host);
        net->runSession(session);
        auto outputTensor = net->getSessionOutput(session, "output");
        MNN::Tensor tensor_scores_host(outputTensor, outputTensor->getDimensionType());
        // 拷贝数据
        outputTensor->copyToHostTensor(&tensor_scores_host);
        std::vector<int> shape = tensor_scores_host.shape();

        // tensor_scores_host.print();
        // 获取host指针
        float *host0 = tensor_scores_host.host<float>();
        // void *host0 = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType());
        // float* host0 = (float*)tensor_scores_host.buffer().host;
        for (int i = 0; i < 100; i++)
        {
            std::cout << host0[i] << "[" << inputData[i] << "]"
                      << " ";
        }
        std::cout << std::endl;

        // 根据维度类型和数据形状，遍历host指针
        // if (dimType == MNN::Tensor::TENSORFLOW) {
        //     // 假设输出tensor的形状是[1, 10]
        //     for (int i = 0; i < 10; i++) {
        //         // 打印host指针的值
        //         printf("score of %d: %f\n", i, host[i]);
        //     }
        // }

        // 打印形状向量
        // printf("The shape of host is: [");
        // for (int i = 0; i < shape.size(); i++)
        // {
        //     printf("%d", shape[i]);
        //     if (i < shape.size() - 1)
        //     {
        //         printf(", ");
        //     }
        // }
        // printf("]\n");

        auto channel = tensor_scores_host.channel(); // 18144
        auto height = tensor_scores_host.height();   // 6
        auto width = tensor_scores_host.width();     // 1
        const float conf_thres = 0.25;
        const float nms_threshold = 0.45;

        const short num_class = height - 5;
        std::vector<Object> l;
        std::cout << channel << " " << width << " " << height << " " << num_class << std::endl;
        for (int c_index = 0; c_index < channel; c_index++)
        {
            float *feat = (float *)host0 + c_index * width * height;

            // find class index with max class score
            int class_index = 0;
            float class_score = -FLT_MAX;

            for (int k = 0; k < num_class; k++)
            {
                // float score = *(feat + 5 + k);
                float score = feat[5 + k];
                // std::cout << "---" << score << " " << feat << " " << ((float *)host0)[1000] << std::endl;
                if (score > class_score)
                {
                    class_index = k;
                    class_score = score;
                }
            }
            float box_score = *(feat + 4);
            float confidence = box_score * class_score;
            // std::cout << "---" << confidence << " " << feat << " " << class_index << std::endl;
            if (confidence < conf_thres)
                continue;

            float center_x = *(feat);
            float center_y = *(feat + 1);
            float width = *(feat + 2);
            float height = *(feat + 3);

            float y0 = center_x - width / 2;
            float y1 = center_y - height / 2;

            Object obj;
            obj.x = y0;
            obj.y = y1;
            obj.x1 = center_x + width / 2;
            obj.y1 = center_y + height / 2;

            obj.width = width;
            obj.height = height;
            obj.label = class_index;
            obj.prob = confidence;

            l.push_back(obj);
            // std::cout << "---" << std::endl;
        }

        // 对vector排序
        std::sort(l.begin(), l.end(), compare_by_prob);
        int count = l.size();
        std::cout << count << std::endl;

        // host = outputTensor->map(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType());
        // outputTensor->unmap(MNN::Tensor::MAP_TENSOR_READ, outputTensor->getDimensionType(), host);
        // int offset = 100;
        // std::cout << host0[offset] << " " << ((float *)host)[offset] << std::endl;
        // std::cout << &host0[0] << " " << &((float *)host)[0] << std::endl;

        auto time = (float)_t.durationInUs() / 1000.0f;
        costs.push_back(time);
    }
    displayStats(costs);

    stbi_image_free(pixel);
}