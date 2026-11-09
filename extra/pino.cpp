#include "pino.h"
#include "pino_inference.h"
#include <iostream>
#include <iomanip>

int main() {
    SDT::PINOInference pino;

    std::wstring model_path = L"conjunction_pino.onnx";
    std::string norm_path = "normalization_params.npz";

    if (!pino.loadModel(model_path, norm_path)) {
        return 1;
    }

    double miss_distance_m = 500.0;
    double rel_speed_m_s = 7500.0;
    double sigma_primary_m = 100.0;
    double sigma_secondary_m = 100.0;
    double hbr_m = 5.0;

    double pc_pred = pino.predict(
        miss_distance_m,
        rel_speed_m_s,
        sigma_primary_m,
        sigma_secondary_m,
        hbr_m
    );

    std::cout << std::scientific << std::setprecision(6);
    std::cout << pc_pred << std::endl;

    return 0;
}