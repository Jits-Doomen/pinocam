#include "pino_inference.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace SDT {


    PINOInference::PINOInference()

        : env_(
            ORT_LOGGING_LEVEL_WARNING,
            "PINO_Inference"
        ),

        memoryInfo_(
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault
            )
        )

    {

        sessionOptions_.SetIntraOpNumThreads(1);

        sessionOptions_.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );
    }


    PINOInference::~PINOInference() = default;


    bool PINOInference::isLoaded() const {

        return session_ != nullptr;
    }


    bool PINOInference::loadModel(
        const std::wstring& modelPath,
        const std::string& normParamsPath
    ) {

        try {

            // ------------------------------------------------
            // 1. Load ONNX model
            // ------------------------------------------------

            session_ = std::make_unique<Ort::Session>(
                env_,
                modelPath.c_str(),
                sessionOptions_
            );


            // ------------------------------------------------
            // 2. Load normalization parameters
            // ------------------------------------------------

            std::ifstream file(normParamsPath);

            if (!file.is_open()) {

                throw std::runtime_error(
                    "Could not open normalization file: "
                    + normParamsPath
                );
            }

            mean_.resize(5);
            std_.resize(5);

            // First line: means.
            for (int i = 0; i < 5; ++i) {

                if (!(file >> mean_[i])) {

                    throw std::runtime_error(
                        "Could not read normalization means."
                    );
                }
            }

            // Second line: standard deviations.
            for (int i = 0; i < 5; ++i) {

                if (!(file >> std_[i])) {

                    throw std::runtime_error(
                        "Could not read normalization standard deviations."
                    );
                }
            }

            for (int i = 0; i < 5; ++i) {

                if (
                    !std::isfinite(mean_[i])
                    || !std::isfinite(std_[i])
                    || std_[i] <= 0.0f
                ) {

                    throw std::runtime_error(
                        "Invalid normalization parameter."
                    );
                }
            }


            // ------------------------------------------------
            // 3. Read ONNX input/output names
            // ------------------------------------------------

            Ort::AllocatorWithDefaultOptions allocator;

            if (
                session_->GetInputCount() != 1
                || session_->GetOutputCount() != 1
            ) {

                throw std::runtime_error(
                    "Expected one ONNX input and one ONNX output."
                );
            }

            auto inputName = session_->GetInputNameAllocated(
                0,
                allocator
            );

            auto outputName = session_->GetOutputNameAllocated(
                0,
                allocator
            );

            inputName_ = inputName.get();
            outputName_ = outputName.get();


            // ------------------------------------------------
            // 4. Validate model input shape and type
            // ------------------------------------------------

            auto inputInfo =
                session_->GetInputTypeInfo(0);

            auto inputTensorInfo =
                inputInfo.GetTensorTypeAndShapeInfo();

            auto inputShape =
                inputTensorInfo.GetShape();

            if (
                inputShape.size() != 2
                || (inputShape[1] != 5 && inputShape[1] != -1)
            ) {

                throw std::runtime_error(
                    "Expected ONNX input shape [batch, 5]."
                );
            }

            if (
                inputTensorInfo.GetElementType()
                != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ) {

                throw std::runtime_error(
                    "ONNX input must use float32."
                );
            }


            // ------------------------------------------------
            // 5. Validate model output type
            // ------------------------------------------------

            auto outputInfo =
                session_->GetOutputTypeInfo(0);

            auto outputTensorInfo =
                outputInfo.GetTensorTypeAndShapeInfo();

            if (
                outputTensorInfo.GetElementType()
                != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ) {

                throw std::runtime_error(
                    "ONNX output must use float32."
                );
            }

            std::cout
                << "PINO model loaded successfully.\n"
                << "Input: " << inputName_ << '\n'
                << "Output: " << outputName_ << '\n';

            return true;

        }

        catch (const Ort::Exception& e) {

            std::cerr
                << "ONNX Runtime error: "
                << e.what()
                << '\n';

            session_.reset();

            return false;
        }

        catch (const std::exception& e) {

            std::cerr
                << "PINO loading error: "
                << e.what()
                << '\n';

            session_.reset();

            return false;
        }
    }


    // ========================================================
    // Prediction
    // ========================================================

    double PINOInference::predict(
        double missDistanceM,
        double relSpeedMS,
        double sigmaPrimaryM,
        double sigmaSecondaryM,
        double hbrM
    ) {

        const double invalid =
            std::numeric_limits<double>::quiet_NaN();

        if (!session_) {
            return invalid;
        }


        // ----------------------------------------------------
        // 1. Raw input features
        // ----------------------------------------------------

        const double rawValues[5] = {
            missDistanceM,
            relSpeedMS,
            sigmaPrimaryM,
            sigmaSecondaryM,
            hbrM
        };


        // ----------------------------------------------------
        // 2. Validate inputs
        // ----------------------------------------------------

        for (double value : rawValues) {

            if (
                !std::isfinite(value)
                || value < 0.0
            ) {

                return invalid;
            }
        }


        // ----------------------------------------------------
        // 3. Normalize inputs
        // ----------------------------------------------------

        std::vector<float> inputValues(5);

        for (int i = 0; i < 5; ++i) {

            inputValues[i] = static_cast<float>(
                (rawValues[i] - mean_[i]) / std_[i]
            );

            if (!std::isfinite(inputValues[i])) {
                return invalid;
            }
        }


        // ----------------------------------------------------
        // 4. Create ONNX input tensor
        // ----------------------------------------------------

        const int64_t inputShape[2] = {
            1,
            5
        };

        try {

            Ort::Value inputTensor =
                Ort::Value::CreateTensor<float>(
                    memoryInfo_,
                    inputValues.data(),
                    inputValues.size(),
                    inputShape,
                    2
                );


            // ------------------------------------------------
            // 5. Run inference
            // ------------------------------------------------

            const char* inputNames[] = {
                inputName_.c_str()
            };

            const char* outputNames[] = {
                outputName_.c_str()
            };

            auto outputTensors = session_->Run(
                Ort::RunOptions{ nullptr },
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1
            );


            // ------------------------------------------------
            // 6. Validate output tensor
            // ------------------------------------------------

            if (
                outputTensors.empty()
                || !outputTensors[0].IsTensor()
            ) {

                return invalid;
            }

            auto outputInfo =
                outputTensors[0].GetTensorTypeAndShapeInfo();

            if (
                outputInfo.GetElementCount() != 1
                || outputInfo.GetElementType()
                    != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT
            ) {

                return invalid;
            }


            // ------------------------------------------------
            // 7. Convert log10(Pc) into Pc
            // ------------------------------------------------

            const float log10Pc =
                outputTensors[0].GetTensorData<float>()[0];

            if (
                !std::isfinite(log10Pc)
                || log10Pc > 0.0f
            ) {

                return invalid;
            }

            const double pc = std::pow(
                10.0,
                static_cast<double>(log10Pc)
            );


            // ------------------------------------------------
            // 8. Final probability validation
            // ------------------------------------------------

            if (
                !std::isfinite(pc)
                || pc < 0.0
                || pc > 1.0
            ) {

                return invalid;
            }

            return pc;

        }

        catch (const Ort::Exception& e) {

            std::cerr
                << "PINO inference error: "
                << e.what()
                << '\n';

            return invalid;
        }
    }

}