#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"
#include "json.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

using namespace nvinfer1;

static Logger gLogger;
const static int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw,
                std::string& img_dir) {
    if (argc < 4)
        return false;
    if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
        wts = std::string(argv[2]);
        engine = std::string(argv[3]);
        auto net = std::string(argv[4]);
        if (net[0] == 'n') {
            gd = 0.33;
            gw = 0.25;
        } else if (net[0] == 's') {
            gd = 0.33;
            gw = 0.50;
        } else if (net[0] == 'm') {
            gd = 0.67;
            gw = 0.75;
        } else if (net[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
        } else if (net[0] == 'x') {
            gd = 1.33;
            gw = 1.25;
        } else if (net[0] == 'c' && argc == 7) {
            gd = atof(argv[5]);
            gw = atof(argv[6]);
        } else {
            return false;
        }
        if (net.size() == 2 && net[1] == '6') {
            is_p6 = true;
        }
    } else if (std::string(argv[1]) == "-d" && argc == 4) {
        engine = std::string(argv[2]);
        img_dir = std::string(argv[3]);
    } else {
        return false;
    }
    return true;
}

void prepare_buffers(ICudaEngine* engine, float** gpu_input_buffer, float** gpu_output_buffer,
                     float** cpu_output_buffer) {
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine->getBindingIndex(kInputTensorName);
    const int outputIndex = engine->getBindingIndex(kOutputTensorName);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)gpu_input_buffer, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)gpu_output_buffer, kBatchSize * kOutputSize * sizeof(float)));

    *cpu_output_buffer = new float[kBatchSize * kOutputSize];
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** gpu_buffers, float* output, int batchsize) {
    context.enqueue(batchsize, gpu_buffers, stream, nullptr);
    CUDA_CHECK(cudaMemcpyAsync(output, gpu_buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                               stream));
    cudaStreamSynchronize(stream);
}

void serialize_engine(unsigned int max_batchsize, bool& is_p6, float& gd, float& gw, std::string& wts_name,
                      std::string& engine_name) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = nullptr;
    if (is_p6) {
        engine = build_det_p6_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    } else {
        engine = build_det_engine(max_batchsize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
    }
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* serialized_engine = engine->serialize();
    assert(serialized_engine != nullptr);

    // Save engine to file
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cerr << "Could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    // Close everything down
    engine->destroy();
    config->destroy();
    serialized_engine->destroy();
    builder->destroy();
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

std::vector<int> class_id_mapping = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90};

int main(int argc, char** argv) {
    cudaSetDevice(kGpuId);

    std::string wts_name = "";
    std::string engine_name = "";
    bool is_p6 = false;
    float gd = 0.0f, gw = 0.0f;
    std::string img_dir;
    nlohmann::json detections_json = nlohmann::json::array(); // 创建一个空的 JSON 数组

    if (!parse_args(argc, argv, wts_name, engine_name, is_p6, gd, gw, img_dir)) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov5_det -s [.wts] [.engine] [n/s/m/l/x/n6/s6/m6/l6/x6 or c/c6 gd gw]  // serialize model to "
                     "plan file"
                  << std::endl;
        std::cerr << "./yolov5_det -d [.engine] ../images  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // Create a model using the API directly and serialize it to a file
    if (!wts_name.empty()) {
        serialize_engine(kBatchSize, is_p6, gd, gw, wts_name, engine_name);
        return 0;
    }

    // Deserialize the engine from file
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;
    deserialize_engine(engine_name, &runtime, &engine, &context);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Init CUDA preprocessing
    cuda_preprocess_init(kMaxInputImageSize);

    // Prepare cpu and gpu buffers
    float* gpu_buffers[2];
    float* cpu_output_buffer = nullptr;
    prepare_buffers(engine, &gpu_buffers[0], &gpu_buffers[1], &cpu_output_buffer);

    // Read images from directory
    std::vector<std::string> file_names;
    if (read_files_in_dir(img_dir.c_str(), file_names) < 0) {
        std::cerr << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // 在循环开始前初始化累积时间变量
    std::chrono::microseconds total_preprocess_time(0);
    std::chrono::microseconds total_inference_time(0);
    std::chrono::microseconds total_nms_time(0);
    size_t total_images = 0; // 用于累积处理的图片总数

    // batch predict
    for (size_t i = 0; i < file_names.size(); i += kBatchSize) {
        // 获取实际批次大小（处理最后一个不完整批次的情况）
        size_t batch_size = std::min(file_names.size() - i, static_cast<size_t>(kBatchSize));

        // Get a batch of images
        std::vector<cv::Mat> img_batch;
        std::vector<std::string> img_name_batch;
        for (size_t j = i; j < i + kBatchSize && j < file_names.size(); j++) {
            cv::Mat img = cv::imread(img_dir + "/" + file_names[j]);
            img_batch.push_back(img);
            img_name_batch.push_back(file_names[j]);
        }

        // Preprocess
        auto n0_start = std::chrono::system_clock::now();
        cuda_batch_preprocess(img_batch, gpu_buffers[0], kInputW, kInputH, stream);
        auto n0_end = std::chrono::system_clock::now();
        total_preprocess_time += std::chrono::duration_cast<std::chrono::microseconds>(n0_end - n0_start);

        // Run inference
        auto n1_start = std::chrono::system_clock::now();
        infer(*context, stream, (void**)gpu_buffers, cpu_output_buffer, kBatchSize);
        auto n1_end = std::chrono::system_clock::now();
        total_inference_time += std::chrono::duration_cast<std::chrono::microseconds>(n1_end - n1_start);

        // NMS
        std::vector<std::vector<Detection>> res_batch;
        auto n2_start = std::chrono::system_clock::now();
        batch_nms(res_batch, cpu_output_buffer, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
        auto n2_end = std::chrono::system_clock::now();
        total_nms_time += std::chrono::duration_cast<std::chrono::microseconds>(n2_end - n2_start);

        // 更新处理的图片总数
        total_images += batch_size;

        // 存入json数组
        for (size_t k = 0; k < res_batch.size(); k++) {
            auto& detections = res_batch[k];
            for (size_t j = 0; j < detections.size(); j++) {
                auto& det = detections[j];
                auto rect_values = get_rect_values(img_batch[k], det.bbox);
                
                std::string filename = img_name_batch[k];
                size_t last_dot_pos = filename.find_last_of(".");
                std::string filename_without_extension = filename.substr(0, last_dot_pos);
                filename_without_extension.erase(0, filename_without_extension.find_first_not_of('0'));
                int image_id_int = std::stoi(filename_without_extension); // 现在这是整数形式的 image_id
                
                nlohmann::json j_detection = {
                    {"image_id", image_id_int},
                    {"category_id", class_id_mapping[det.class_id]},
                    {"bbox", {std::get<0>(rect_values), std::get<1>(rect_values), std::get<2>(rect_values), std::get<3>(rect_values)}},
                    {"score", det.conf}
                };
                
                detections_json.push_back(j_detection); // 直接将检测结果添加到 JSON 数组中
            }
        }

        // // Draw bounding boxes
        // draw_bbox(img_batch, res_batch);

        // Save images
        // for (size_t j = 0; j < img_batch.size(); j++) {
        //     cv::imwrite("output/" + img_name_batch[j], img_batch[j]);
        // }
    }
    std::ofstream file("detections.json");
    file << detections_json.dump(4); // 使用4个空格的缩进美化JSON输出
    file.close();

    // 计算并输出每张图片的平均处理时间
    auto avg_preprocess_time_per_image = total_preprocess_time.count() / total_images;
    auto avg_inference_time_per_image = total_inference_time.count() / total_images;
    auto avg_nms_time_per_image = total_nms_time.count() / total_images;
    // 使用1秒（1000000微秒）除以每张图片的处理时间来计算FPS
    double fps = 1000000.0 / (avg_preprocess_time_per_image + avg_inference_time_per_image + avg_nms_time_per_image);

    std::cout << "Average Preprocess Time per Image: " << avg_preprocess_time_per_image << "us" << std::endl;
    std::cout << "Average Inference Time per Image: " << avg_inference_time_per_image << "us" << std::endl;
    std::cout << "Average NMS Time per Image: " << avg_nms_time_per_image << "us" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Total Images: " << total_images << std::endl;
    std::cout << "Total Time: " << total_preprocess_time.count() + total_inference_time.count() + total_nms_time.count() << "us" << std::endl;

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(gpu_buffers[0]));
    CUDA_CHECK(cudaFree(gpu_buffers[1]));
    delete[] cpu_output_buffer;
    cuda_preprocess_destroy();
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    // std::cout << "\nOutput:\n\n";
    // for (unsigned int i = 0; i < kOutputSize; i++) {
    //   std::cout << prob[i] << ", ";
    //   if (i % 10 == 0) std::cout << std::endl;
    // }
    // std::cout << std::endl;

    return 0;
}
