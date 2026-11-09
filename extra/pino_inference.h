#pragma once

#include <string>
#include <vector>
#include <memory>

#include <onnxruntime_cxx_api.h>

namespace SDT {

    class PINOInference {

    public:

        PINOInference();
        ~PINOInference();

        bool loadModel(
            const std::wstring& modelPath,
            const std::string& normParamsPath
        );

        double predict(
            double missDistanceM,
            double relSpeedMS,
            double sigmaPrimaryM,
            double sigmaSecondaryM,
            double hbrM
        );

        bool isLoaded() const;

    private:

        Ort::Env env_;

        Ort::SessionOptions sessionOptions_;

        std::unique_ptr<Ort::Session> session_;

        Ort::MemoryInfo memoryInfo_;

        std::vector<float> mean_;
        std::vector<float> std_;

        std::string inputName_;
        std::string outputName_;

    };

}