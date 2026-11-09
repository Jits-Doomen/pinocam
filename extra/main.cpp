#include "pino_inference.h"

#include <iostream>
#include <iomanip>
#include <cmath>

int main() {

    SDT::PINOInference pino;

    const std::wstring modelPath = L"conjunction_pino.onnx";
    const std::string normPath = "normalization_params.txt";

    if (!pino.loadModel(modelPath, normPath)) {
        std::cerr << "Failed to initialize PINO.\n";
        return 1;
    }

    // Illustrative test inputs.
    // These are not measurements from a real conjunction.
    const double missDistanceM = 500.0;
    const double relSpeedMS = 7500.0;
    const double sigmaPrimaryM = 100.0;
    const double sigmaSecondaryM = 100.0;
    const double hbrM = 5.0;

    const double pc = pino.predict(
        missDistanceM,
        relSpeedMS,
        sigmaPrimaryM,
        sigmaSecondaryM,
        hbrM
    );

    if (!std::isfinite(pc)) {

        std::cerr << "Invalid PINO prediction.\n";
        return 2;
    }

    std::cout << std::scientific
              << std::setprecision(8);

    std::cout << "PINO inference test\n";
    std::cout << "-------------------\n";

    std::cout << "Miss distance: "
              << missDistanceM << " m\n";

    std::cout << "Relative speed: "
              << relSpeedMS << " m/s\n";

    std::cout << "Primary sigma: "
              << sigmaPrimaryM << " m\n";

    std::cout << "Secondary sigma: "
              << sigmaSecondaryM << " m\n";

    std::cout << "Hard-body radius: "
              << hbrM << " m\n";

    std::cout << "Predicted Pc: "
              << pc << '\n';

    return 0;
}