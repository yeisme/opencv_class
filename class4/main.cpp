#include <filesystem>
#include <opencv2/opencv.hpp>
#include <print>
#include <string>

/**
 * @brief Roberts算子边缘检测
 *
 * Roberts算子是一种简单的边缘检测算子，使用2x2卷积核来检测图像中的边缘。
 * 它通过计算相邻像素的差值来检测边缘，对噪声比较敏感但计算速度快。
 *
 * @param input 输入的灰度图像
 * @param horizontal 水平方向的Roberts算子响应结果
 * @param vertical 垂直方向的Roberts算子响应结果
 * @param euclidean 使用欧几里德距离计算的梯度幅值
 * @param manhattan 使用曼哈顿距离计算的梯度幅值
 *
 * @note 所有输出参数都是CV_32F类型的浮点图像
 * @see prewittEdgeDetection, sobelEdgeDetection
 */
void robertsEdgeDetection(const cv::Mat &input, cv::Mat &horizontal, cv::Mat &vertical, cv::Mat &euclidean,
                          cv::Mat &manhattan)
{
    // Roberts算子核
    cv::Mat rh = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0); // 水平Roberts算子
    cv::Mat rv = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1); // 垂直Roberts算子

    // 转换为浮点型进行计算
    cv::Mat float_input;
    input.convertTo(float_input, CV_32F);

    // 应用Roberts算子
    cv::filter2D(float_input, horizontal, CV_32F, rh);
    cv::filter2D(float_input, vertical, CV_32F, rv);

    // 计算梯度模
    // 欧几里德距离 (L2范数)
    cv::sqrt(horizontal.mul(horizontal) + vertical.mul(vertical), euclidean);

    // 街区距离 (L1范数)
    manhattan = cv::abs(horizontal) + cv::abs(vertical);
}

/**
 * @brief Prewitt算子边缘检测
 *
 * Prewitt算子是一种经典的边缘检测算子，使用3x3卷积核来检测图像中的边缘。
 * 相比Roberts算子，Prewitt算子对噪声的抵抗能力更强，边缘检测效果更平滑。
 *
 * @param input 输入的灰度图像
 * @param horizontal 水平方向的Prewitt算子响应结果
 * @param vertical 垂直方向的Prewitt算子响应结果
 * @param euclidean 使用欧几里德距离计算的梯度幅值
 * @param manhattan 使用曼哈顿距离计算的梯度幅值
 *
 * @note 所有输出参数都是CV_32F类型的浮点图像
 * @see robertsEdgeDetection, sobelEdgeDetection
 */
void prewittEdgeDetection(const cv::Mat &input, cv::Mat &horizontal, cv::Mat &vertical, cv::Mat &euclidean,
                          cv::Mat &manhattan)
{
    // Prewitt算子核
    cv::Mat ph = (cv::Mat_<float>(3, 3) << -1, -1, -1, 0, 0, 0, 1, 1, 1); // 水平Prewitt算子
    cv::Mat pv = (cv::Mat_<float>(3, 3) << -1, 0, 1, -1, 0, 1, -1, 0, 1); // 垂直Prewitt算子

    // 转换为浮点型进行计算
    cv::Mat float_input;
    input.convertTo(float_input, CV_32F);

    // 应用Prewitt算子
    cv::filter2D(float_input, horizontal, CV_32F, ph);
    cv::filter2D(float_input, vertical, CV_32F, pv);

    // 计算梯度模
    // 欧几里德距离 (L2范数)
    cv::sqrt(horizontal.mul(horizontal) + vertical.mul(vertical), euclidean);

    // 街区距离 (L1范数)
    manhattan = cv::abs(horizontal) + cv::abs(vertical);
}

/**
 * @brief Sobel算子边缘检测
 *
 * Sobel算子是最常用的边缘检测算子之一，使用3x3卷积核并对中心像素给予更高权重。
 * 它在噪声抑制和边缘检测效果之间提供了良好的平衡，是实际应用中的首选算子。
 *
 * @param input 输入的灰度图像
 * @param horizontal 水平方向的Sobel算子响应结果
 * @param vertical 垂直方向的Sobel算子响应结果
 * @param euclidean 使用欧几里德距离计算的梯度幅值
 * @param manhattan 使用曼哈顿距离计算的梯度幅值
 *
 * @note 所有输出参数都是CV_32F类型的浮点图像
 * @note Sobel算子的响应通常比Roberts和Prewitt算子更强
 * @see robertsEdgeDetection, prewittEdgeDetection
 */
void sobelEdgeDetection(const cv::Mat &input, cv::Mat &horizontal, cv::Mat &vertical, cv::Mat &euclidean,
                        cv::Mat &manhattan)
{
    // Sobel算子核
    cv::Mat sh = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1); // 水平Sobel算子
    cv::Mat sv = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); // 垂直Sobel算子

    // 转换为浮点型进行计算
    cv::Mat float_input;
    input.convertTo(float_input, CV_32F);

    // 应用Sobel算子
    cv::filter2D(float_input, horizontal, CV_32F, sh);
    cv::filter2D(float_input, vertical, CV_32F, sv);

    // 计算梯度模
    // 欧几里德距离 (L2范数)
    cv::sqrt(horizontal.mul(horizontal) + vertical.mul(vertical), euclidean);

    // 街区距离 (L1范数)
    manhattan = cv::abs(horizontal) + cv::abs(vertical);
}

/**
 * @brief 添加高斯噪声
 *
 * 此函数向输入图像添加零均值的高斯噪声，用于测试边缘检测算法在噪声环境下的性能。
 * 生成的噪声图像可以用来评估不同边缘检测算子的噪声鲁棒性。
 *
 * @param input 输入的原始图像
 * @param mean 高斯噪声的均值，默认为0（零均值噪声）
 * @param stddev 高斯噪声的标准差，控制噪声强度，默认为25
 *
 * @return cv::Mat 添加噪声后的图像，数据类型为CV_8U
 *
 * @note 函数内部会自动处理数据类型转换和像素值截断
 * @warning 过大的stddev值可能导致图像质量严重下降
 *
 * @example
 * @code
 * cv::Mat noisy = addGaussianNoise(original_image, 0, 30);
 * @endcode
 */
cv::Mat addGaussianNoise(const cv::Mat &input, double mean = 0, double stddev = 25)
{
    cv::Mat noise = cv::Mat::zeros(input.size(), CV_32F);
    cv::randn(noise, cv::Scalar(mean), cv::Scalar(stddev));

    cv::Mat float_input;
    input.convertTo(float_input, CV_32F);

    cv::Mat noisy_image = float_input + noise;
    cv::Mat result;
    noisy_image.convertTo(result, CV_8U);

    return result;
}

/**
 * @brief LoG (拉普拉斯-高斯)算子边缘检测
 *
 * LoG算子首先对图像进行高斯滤波以减少噪声，然后应用拉普拉斯算子检测边缘。
 * 这种组合方法在噪声抑制和边缘检测之间提供了很好的平衡，特别适合处理噪声图像。
 *
 * @param input 输入的灰度图像
 * @param output LoG算子的响应结果，直接输出边缘强度（无需额外计算梯度模）
 * @param ksize 高斯滤波核的大小，必须为奇数，默认为5
 * @param sigma 高斯滤波的标准差，控制平滑程度，默认为1.0
 *
 * @note 输出图像为CV_32F类型，已取绝对值便于显示
 * @note 较小的sigma值检测细节边缘，较大的sigma值检测粗边缘
 * @note 与其他算子不同，LoG直接输出边缘强度，无需计算梯度模
 *
 * @warning ksize必须为奇数且大于1
 * @warning sigma值过小可能导致噪声增强，过大可能丢失细节
 *
 * @see robertsEdgeDetection, prewittEdgeDetection, sobelEdgeDetection
 *
 * @example
 * @code
 * cv::Mat log_result;
 * logEdgeDetection(image, log_result, 7, 1.5);  // 使用7x7核，sigma=1.5
 * @endcode
 */
void logEdgeDetection(const cv::Mat &input, cv::Mat &output, int ksize = 5, double sigma = 1.0)
{
    // 转换为浮点型进行计算
    cv::Mat float_input;
    input.convertTo(float_input, CV_32F);

    // 步骤1: 高斯滤波
    cv::Mat gaussian_filtered;
    cv::GaussianBlur(float_input, gaussian_filtered, cv::Size(ksize, ksize), sigma);

    // 步骤2: 拉普拉斯算子
    cv::Mat laplacian_kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 4, -1, 0, -1, 0);
    cv::filter2D(gaussian_filtered, output, CV_32F, laplacian_kernel);

    // 取绝对值以便显示
    output = cv::abs(output);
}

/**
 * @brief 主函数 - 边缘检测算法的完整演示
 *
 * 本程序实现了四种经典边缘检测算法的完整流程：
 * 1. Roberts算子（2x2）- 简单快速，对噪声敏感
 * 2. Prewitt算子（3x3）- 平滑效果好，噪声抗性中等
 * 3. Sobel算子（3x3）- 实用性强，广泛应用
 * 4. LoG算子 - 噪声抑制能力强，适合噪声环境
 *
 * 对每种算子都进行以下处理：
 * - 水平和垂直方向的边缘检测
 * - 欧几里德距离和曼哈顿距离的梯度计算
 * - 固定阈值和OTSU自适应阈值的二值化
 * - 正常显示和白底黑线的图像格式
 * - 原始图像和噪声图像的对比处理
 *
 * @return int 程序退出状态码，0表示成功，-1表示图像加载失败
 *
 * @note 程序会生成大量图像文件，建议在空目录中运行
 * @note 所有生成的图像文件都使用中文命名，便于识别
 * @note OTSU方法可以自适应确定最佳二值化阈值
 *
 * @warning 确保img/img1.png文件存在且可读
 * @warning 程序会覆盖同名的输出文件
 */
int main()
{
    std::string image_path = "img/img1.png";
    cv::Mat original_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

    if (original_image.empty())
    {
        std::print("错误: 无法加载图像 '{}'. 请检查路径是否正确。\n", image_path);
        return -1;
    }

    // 创建输出目录
    std::filesystem::create_directories("output/原始图像");
    std::filesystem::create_directories("output/Roberts算子");
    std::filesystem::create_directories("output/Prewitt算子");
    std::filesystem::create_directories("output/Sobel算子");
    std::filesystem::create_directories("output/LoG算子");
    std::filesystem::create_directories("output/OTSU二值化");

    // 保存原始图像
    cv::imwrite("output/原始图像/原始灰度图像.png", original_image);

    // Roberts边缘检测
    cv::Mat horizontal, vertical, euclidean, manhattan;
    robertsEdgeDetection(original_image, horizontal, vertical, euclidean, manhattan);

    // 转换为8位图像用于保存
    cv::Mat horizontal_8u, vertical_8u, euclidean_8u, manhattan_8u;
    cv::convertScaleAbs(horizontal, horizontal_8u);
    cv::convertScaleAbs(vertical, vertical_8u);
    cv::convertScaleAbs(euclidean, euclidean_8u);
    cv::convertScaleAbs(manhattan, manhattan_8u);

    // 保存Roberts检测结果
    cv::imwrite("output/Roberts算子/Roberts水平边缘.png", horizontal_8u);
    cv::imwrite("output/Roberts算子/Roberts垂直边缘.png", vertical_8u);
    cv::imwrite("output/Roberts算子/欧几里德梯度.png", euclidean_8u);
    cv::imwrite("output/Roberts算子/曼哈顿梯度.png", manhattan_8u);

    // 二值化处理
    cv::Mat binary_euclidean, binary_manhattan;
    double euclidean_threshold = 30;
    double manhattan_threshold = 40;

    cv::threshold(euclidean_8u, binary_euclidean, euclidean_threshold, 255, cv::THRESH_BINARY);
    cv::threshold(manhattan_8u, binary_manhattan, manhattan_threshold, 255, cv::THRESH_BINARY);

    cv::imwrite("output/Roberts算子/二值化欧几里德.png", binary_euclidean);
    cv::imwrite("output/Roberts算子/二值化曼哈顿.png", binary_manhattan);

    // 转换为白底黑线条
    cv::Mat inverted_euclidean, inverted_manhattan;
    cv::bitwise_not(binary_euclidean, inverted_euclidean);
    cv::bitwise_not(binary_manhattan, inverted_manhattan);

    cv::imwrite("output/Roberts算子/白底黑线欧几里德.png", inverted_euclidean);
    cv::imwrite("output/Roberts算子/白底黑线曼哈顿.png", inverted_manhattan);

    std::print("Roberts原始图像处理完成...\n");

    // 添加高斯噪声
    cv::Mat noisy_image = addGaussianNoise(original_image);
    cv::imwrite("output/原始图像/噪声图像.png", noisy_image);

    std::print("处理噪声图像...\n");

    // 对噪声图像进行Roberts边缘检测
    cv::Mat noisy_horizontal, noisy_vertical, noisy_euclidean, noisy_manhattan;
    robertsEdgeDetection(noisy_image, noisy_horizontal, noisy_vertical, noisy_euclidean, noisy_manhattan);

    // 转换为8位图像
    cv::Mat noisy_horizontal_8u, noisy_vertical_8u, noisy_euclidean_8u, noisy_manhattan_8u;
    cv::convertScaleAbs(noisy_horizontal, noisy_horizontal_8u);
    cv::convertScaleAbs(noisy_vertical, noisy_vertical_8u);
    cv::convertScaleAbs(noisy_euclidean, noisy_euclidean_8u);
    cv::convertScaleAbs(noisy_manhattan, noisy_manhattan_8u);

    // 保存噪声图像的Roberts检测结果
    cv::imwrite("output/Roberts算子/噪声Roberts水平边缘.png", noisy_horizontal_8u);
    cv::imwrite("output/Roberts算子/噪声Roberts垂直边缘.png", noisy_vertical_8u);
    cv::imwrite("output/Roberts算子/噪声欧几里德梯度.png", noisy_euclidean_8u);
    cv::imwrite("output/Roberts算子/噪声曼哈顿梯度.png", noisy_manhattan_8u);

    // 噪声图像的二值化处理
    cv::Mat noisy_binary_euclidean, noisy_binary_manhattan;
    cv::threshold(noisy_euclidean_8u, noisy_binary_euclidean, euclidean_threshold + 10, 255, cv::THRESH_BINARY);
    cv::threshold(noisy_manhattan_8u, noisy_binary_manhattan, manhattan_threshold + 10, 255, cv::THRESH_BINARY);

    cv::imwrite("output/Roberts算子/噪声二值化欧几里德.png", noisy_binary_euclidean);
    cv::imwrite("output/Roberts算子/噪声二值化曼哈顿.png", noisy_binary_manhattan);

    // 噪声图像转换为白底黑线条
    cv::Mat noisy_inverted_euclidean, noisy_inverted_manhattan;
    cv::bitwise_not(noisy_binary_euclidean, noisy_inverted_euclidean);
    cv::bitwise_not(noisy_binary_manhattan, noisy_inverted_manhattan);

    cv::imwrite("output/Roberts算子/噪声白底黑线欧几里德.png", noisy_inverted_euclidean);
    cv::imwrite("output/Roberts算子/噪声白底黑线曼哈顿.png", noisy_inverted_manhattan);

    // ========== Prewitt算子边缘检测 ==========
    std::print("\n开始处理Prewitt算子边缘检测...\n");

    // Prewitt边缘检测 - 原始图像
    cv::Mat prewitt_horizontal, prewitt_vertical, prewitt_euclidean, prewitt_manhattan;
    prewittEdgeDetection(original_image, prewitt_horizontal, prewitt_vertical, prewitt_euclidean, prewitt_manhattan);

    // 转换为8位图像用于保存
    cv::Mat prewitt_horizontal_8u, prewitt_vertical_8u, prewitt_euclidean_8u, prewitt_manhattan_8u;
    cv::convertScaleAbs(prewitt_horizontal, prewitt_horizontal_8u);
    cv::convertScaleAbs(prewitt_vertical, prewitt_vertical_8u);
    cv::convertScaleAbs(prewitt_euclidean, prewitt_euclidean_8u);
    cv::convertScaleAbs(prewitt_manhattan, prewitt_manhattan_8u);

    // 保存Prewitt检测结果
    cv::imwrite("output/Prewitt算子/Prewitt水平边缘.png", prewitt_horizontal_8u);
    cv::imwrite("output/Prewitt算子/Prewitt垂直边缘.png", prewitt_vertical_8u);
    cv::imwrite("output/Prewitt算子/Prewitt欧几里德梯度.png", prewitt_euclidean_8u);
    cv::imwrite("output/Prewitt算子/Prewitt曼哈顿梯度.png", prewitt_manhattan_8u);

    // Prewitt二值化处理
    cv::Mat prewitt_binary_euclidean, prewitt_binary_manhattan;
    double prewitt_euclidean_threshold = 50;
    double prewitt_manhattan_threshold = 60;

    cv::threshold(prewitt_euclidean_8u, prewitt_binary_euclidean, prewitt_euclidean_threshold, 255, cv::THRESH_BINARY);
    cv::threshold(prewitt_manhattan_8u, prewitt_binary_manhattan, prewitt_manhattan_threshold, 255, cv::THRESH_BINARY);

    cv::imwrite("output/Prewitt算子/Prewitt二值化欧几里德.png", prewitt_binary_euclidean);
    cv::imwrite("output/Prewitt算子/Prewitt二值化曼哈顿.png", prewitt_binary_manhattan);

    // Prewitt转换为白底黑线条
    cv::Mat prewitt_inverted_euclidean, prewitt_inverted_manhattan;
    cv::bitwise_not(prewitt_binary_euclidean, prewitt_inverted_euclidean);
    cv::bitwise_not(prewitt_binary_manhattan, prewitt_inverted_manhattan);

    cv::imwrite("output/Prewitt算子/Prewitt白底黑线欧几里德.png", prewitt_inverted_euclidean);
    cv::imwrite("output/Prewitt算子/Prewitt白底黑线曼哈顿.png", prewitt_inverted_manhattan);

    std::print("Prewitt原始图像处理完成...\n");

    // 对噪声图像进行Prewitt边缘检测
    cv::Mat prewitt_noisy_horizontal, prewitt_noisy_vertical, prewitt_noisy_euclidean, prewitt_noisy_manhattan;
    prewittEdgeDetection(noisy_image, prewitt_noisy_horizontal, prewitt_noisy_vertical, prewitt_noisy_euclidean,
                         prewitt_noisy_manhattan);

    // 转换为8位图像
    cv::Mat prewitt_noisy_horizontal_8u, prewitt_noisy_vertical_8u, prewitt_noisy_euclidean_8u,
        prewitt_noisy_manhattan_8u;
    cv::convertScaleAbs(prewitt_noisy_horizontal, prewitt_noisy_horizontal_8u);
    cv::convertScaleAbs(prewitt_noisy_vertical, prewitt_noisy_vertical_8u);
    cv::convertScaleAbs(prewitt_noisy_euclidean, prewitt_noisy_euclidean_8u);
    cv::convertScaleAbs(prewitt_noisy_manhattan, prewitt_noisy_manhattan_8u);

    // 保存Prewitt噪声图像的检测结果
    cv::imwrite("output/Prewitt算子/噪声Prewitt水平边缘.png", prewitt_noisy_horizontal_8u);
    cv::imwrite("output/Prewitt算子/噪声Prewitt垂直边缘.png", prewitt_noisy_vertical_8u);
    cv::imwrite("output/Prewitt算子/噪声Prewitt欧几里德梯度.png", prewitt_noisy_euclidean_8u);
    cv::imwrite("output/Prewitt算子/噪声Prewitt曼哈顿梯度.png", prewitt_noisy_manhattan_8u);

    // Prewitt噪声图像的二值化处理
    cv::Mat prewitt_noisy_binary_euclidean, prewitt_noisy_binary_manhattan;
    cv::threshold(prewitt_noisy_euclidean_8u, prewitt_noisy_binary_euclidean, prewitt_euclidean_threshold + 15, 255,
                  cv::THRESH_BINARY);
    cv::threshold(prewitt_noisy_manhattan_8u, prewitt_noisy_binary_manhattan, prewitt_manhattan_threshold + 15, 255,
                  cv::THRESH_BINARY);

    cv::imwrite("output/Prewitt算子/噪声Prewitt二值化欧几里德.png", prewitt_noisy_binary_euclidean);
    cv::imwrite("output/Prewitt算子/噪声Prewitt二值化曼哈顿.png", prewitt_noisy_binary_manhattan);

    // Prewitt噪声图像转换为白底黑线条
    cv::Mat prewitt_noisy_inverted_euclidean, prewitt_noisy_inverted_manhattan;
    cv::bitwise_not(prewitt_noisy_binary_euclidean, prewitt_noisy_inverted_euclidean);
    cv::bitwise_not(prewitt_noisy_binary_manhattan, prewitt_noisy_inverted_manhattan);

    cv::imwrite("output/Prewitt算子/噪声Prewitt白底黑线欧几里德.png", prewitt_noisy_inverted_euclidean);
    cv::imwrite("output/Prewitt算子/噪声Prewitt白底黑线曼哈顿.png", prewitt_noisy_inverted_manhattan);

    // ========== Sobel算子边缘检测 ==========
    std::print("\n开始处理Sobel算子边缘检测...\n");

    // Sobel边缘检测 - 原始图像
    cv::Mat sobel_horizontal, sobel_vertical, sobel_euclidean, sobel_manhattan;
    sobelEdgeDetection(original_image, sobel_horizontal, sobel_vertical, sobel_euclidean, sobel_manhattan);

    // 转换为8位图像用于保存
    cv::Mat sobel_horizontal_8u, sobel_vertical_8u, sobel_euclidean_8u, sobel_manhattan_8u;
    cv::convertScaleAbs(sobel_horizontal, sobel_horizontal_8u);
    cv::convertScaleAbs(sobel_vertical, sobel_vertical_8u);
    cv::convertScaleAbs(sobel_euclidean, sobel_euclidean_8u);
    cv::convertScaleAbs(sobel_manhattan, sobel_manhattan_8u);

    // 保存Sobel检测结果
    cv::imwrite("output/Sobel算子/Sobel水平边缘.png", sobel_horizontal_8u);
    cv::imwrite("output/Sobel算子/Sobel垂直边缘.png", sobel_vertical_8u);
    cv::imwrite("output/Sobel算子/Sobel欧几里德梯度.png", sobel_euclidean_8u);
    cv::imwrite("output/Sobel算子/Sobel曼哈顿梯度.png", sobel_manhattan_8u);

    // Sobel二值化处理
    cv::Mat sobel_binary_euclidean, sobel_binary_manhattan;
    double sobel_euclidean_threshold = 80;
    double sobel_manhattan_threshold = 100;

    cv::threshold(sobel_euclidean_8u, sobel_binary_euclidean, sobel_euclidean_threshold, 255, cv::THRESH_BINARY);
    cv::threshold(sobel_manhattan_8u, sobel_binary_manhattan, sobel_manhattan_threshold, 255, cv::THRESH_BINARY);

    cv::imwrite("output/Sobel算子/Sobel二值化欧几里德.png", sobel_binary_euclidean);
    cv::imwrite("output/Sobel算子/Sobel二值化曼哈顿.png", sobel_binary_manhattan);

    // Sobel转换为白底黑线条
    cv::Mat sobel_inverted_euclidean, sobel_inverted_manhattan;
    cv::bitwise_not(sobel_binary_euclidean, sobel_inverted_euclidean);
    cv::bitwise_not(sobel_binary_manhattan, sobel_inverted_manhattan);

    cv::imwrite("output/Sobel算子/Sobel白底黑线欧几里德.png", sobel_inverted_euclidean);
    cv::imwrite("output/Sobel算子/Sobel白底黑线曼哈顿.png", sobel_inverted_manhattan);

    std::print("Sobel原始图像处理完成...\n");

    // 对噪声图像进行Sobel边缘检测
    cv::Mat sobel_noisy_horizontal, sobel_noisy_vertical, sobel_noisy_euclidean, sobel_noisy_manhattan;
    sobelEdgeDetection(noisy_image, sobel_noisy_horizontal, sobel_noisy_vertical, sobel_noisy_euclidean,
                       sobel_noisy_manhattan);

    // 转换为8位图像
    cv::Mat sobel_noisy_horizontal_8u, sobel_noisy_vertical_8u, sobel_noisy_euclidean_8u, sobel_noisy_manhattan_8u;
    cv::convertScaleAbs(sobel_noisy_horizontal, sobel_noisy_horizontal_8u);
    cv::convertScaleAbs(sobel_noisy_vertical, sobel_noisy_vertical_8u);
    cv::convertScaleAbs(sobel_noisy_euclidean, sobel_noisy_euclidean_8u);
    cv::convertScaleAbs(sobel_noisy_manhattan, sobel_noisy_manhattan_8u);

    // 保存Sobel噪声图像的检测结果
    cv::imwrite("output/Sobel算子/噪声Sobel水平边缘.png", sobel_noisy_horizontal_8u);
    cv::imwrite("output/Sobel算子/噪声Sobel垂直边缘.png", sobel_noisy_vertical_8u);
    cv::imwrite("output/Sobel算子/噪声Sobel欧几里德梯度.png", sobel_noisy_euclidean_8u);
    cv::imwrite("output/Sobel算子/噪声Sobel曼哈顿梯度.png", sobel_noisy_manhattan_8u);

    // Sobel噪声图像的二值化处理
    cv::Mat sobel_noisy_binary_euclidean, sobel_noisy_binary_manhattan;
    cv::threshold(sobel_noisy_euclidean_8u, sobel_noisy_binary_euclidean, sobel_euclidean_threshold + 20, 255,
                  cv::THRESH_BINARY);
    cv::threshold(sobel_noisy_manhattan_8u, sobel_noisy_binary_manhattan, sobel_manhattan_threshold + 20, 255,
                  cv::THRESH_BINARY);

    cv::imwrite("output/Sobel算子/噪声Sobel二值化欧几里德.png", sobel_noisy_binary_euclidean);
    cv::imwrite("output/Sobel算子/噪声Sobel二值化曼哈顿.png", sobel_noisy_binary_manhattan);

    // Sobel噪声图像转换为白底黑线条
    cv::Mat sobel_noisy_inverted_euclidean, sobel_noisy_inverted_manhattan;
    cv::bitwise_not(sobel_noisy_binary_euclidean, sobel_noisy_inverted_euclidean);
    cv::bitwise_not(sobel_noisy_binary_manhattan, sobel_noisy_inverted_manhattan);

    cv::imwrite("output/Sobel算子/噪声Sobel白底黑线欧几里德.png", sobel_noisy_inverted_euclidean);
    cv::imwrite("output/Sobel算子/噪声Sobel白底黑线曼哈顿.png", sobel_noisy_inverted_manhattan);

    // ========== LoG算子边缘检测 ==========
    std::print("\n开始处理LoG算子边缘检测...\n");

    // LoG边缘检测 - 原始图像
    cv::Mat log_result;
    logEdgeDetection(original_image, log_result, 5, 1.0);

    // 转换为8位图像用于保存
    cv::Mat log_result_8u;
    cv::convertScaleAbs(log_result, log_result_8u);

    // 保存LoG检测结果
    cv::imwrite("output/LoG算子/LoG边缘检测.png", log_result_8u);

    // LoG二值化处理
    cv::Mat log_binary;
    double log_threshold = 20; // LoG算子通常需要较低的阈值

    cv::threshold(log_result_8u, log_binary, log_threshold, 255, cv::THRESH_BINARY);
    cv::imwrite("output/LoG算子/LoG二值化.png", log_binary);

    // LoG转换为白底黑线条
    cv::Mat log_inverted;
    cv::bitwise_not(log_binary, log_inverted);
    cv::imwrite("output/LoG算子/LoG白底黑线.png", log_inverted);

    std::print("LoG原始图像处理完成...\n");

    // 对噪声图像进行LoG边缘检测
    cv::Mat log_noisy_result;
    logEdgeDetection(noisy_image, log_noisy_result, 7, 1.5); // 使用更大的核和sigma来处理噪声

    // 转换为8位图像
    cv::Mat log_noisy_result_8u;
    cv::convertScaleAbs(log_noisy_result, log_noisy_result_8u);

    // 保存LoG噪声图像的检测结果
    cv::imwrite("output/LoG算子/噪声LoG边缘检测.png", log_noisy_result_8u);

    // LoG噪声图像的二值化处理
    cv::Mat log_noisy_binary;
    cv::threshold(log_noisy_result_8u, log_noisy_binary, log_threshold + 5, 255, cv::THRESH_BINARY);
    cv::imwrite("output/LoG算子/噪声LoG二值化.png", log_noisy_binary);

    // LoG噪声图像转换为白底黑线条
    cv::Mat log_noisy_inverted;
    cv::bitwise_not(log_noisy_binary, log_noisy_inverted);
    cv::imwrite("output/LoG算子/噪声LoG白底黑线.png", log_noisy_inverted);

    // 测试不同参数的LoG算子
    std::print("生成不同参数的LoG处理结果...\n");

    // 小sigma值 - 检测细节边缘
    cv::Mat log_small_sigma;
    logEdgeDetection(original_image, log_small_sigma, 5, 0.5);
    cv::Mat log_small_sigma_8u;
    cv::convertScaleAbs(log_small_sigma, log_small_sigma_8u);
    cv::imwrite("output/LoG算子/LoG小sigma边缘检测.png", log_small_sigma_8u);

    // 大sigma值 - 检测粗边缘
    cv::Mat log_large_sigma;
    logEdgeDetection(original_image, log_large_sigma, 9, 2.0);
    cv::Mat log_large_sigma_8u;
    cv::convertScaleAbs(log_large_sigma, log_large_sigma_8u);
    cv::imwrite("output/LoG算子/LoG大sigma边缘检测.png", log_large_sigma_8u);

    // ========== OTSU方法二值化处理 ==========
    std::print("\n开始使用OTSU方法对所有边缘检测结果进行二值化处理...\n");

    // Roberts算子OTSU二值化
    cv::Mat roberts_otsu_euclidean, roberts_otsu_manhattan;
    cv::threshold(euclidean_8u, roberts_otsu_euclidean, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(manhattan_8u, roberts_otsu_manhattan, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imwrite("output/OTSU二值化/Roberts_OTSU二值化欧几里德.png", roberts_otsu_euclidean);
    cv::imwrite("output/OTSU二值化/Roberts_OTSU二值化曼哈顿.png", roberts_otsu_manhattan);

    // Roberts噪声图像OTSU二值化
    cv::Mat roberts_noisy_otsu_euclidean, roberts_noisy_otsu_manhattan;
    cv::threshold(noisy_euclidean_8u, roberts_noisy_otsu_euclidean, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(noisy_manhattan_8u, roberts_noisy_otsu_manhattan, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imwrite("output/OTSU二值化/噪声Roberts_OTSU二值化欧几里德.png", roberts_noisy_otsu_euclidean);
    cv::imwrite("output/OTSU二值化/噪声Roberts_OTSU二值化曼哈顿.png", roberts_noisy_otsu_manhattan);

    // Prewitt算子OTSU二值化
    cv::Mat prewitt_otsu_euclidean, prewitt_otsu_manhattan;
    cv::threshold(prewitt_euclidean_8u, prewitt_otsu_euclidean, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(prewitt_manhattan_8u, prewitt_otsu_manhattan, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imwrite("output/OTSU二值化/Prewitt_OTSU二值化欧几里德.png", prewitt_otsu_euclidean);
    cv::imwrite("output/OTSU二值化/Prewitt_OTSU二值化曼哈顿.png", prewitt_otsu_manhattan);

    // Prewitt噪声图像OTSU二值化
    cv::Mat prewitt_noisy_otsu_euclidean, prewitt_noisy_otsu_manhattan;
    cv::threshold(prewitt_noisy_euclidean_8u, prewitt_noisy_otsu_euclidean, 0, 255,
                  cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(prewitt_noisy_manhattan_8u, prewitt_noisy_otsu_manhattan, 0, 255,
                  cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imwrite("output/OTSU二值化/噪声Prewitt_OTSU二值化欧几里德.png", prewitt_noisy_otsu_euclidean);
    cv::imwrite("output/OTSU二值化/噪声Prewitt_OTSU二值化曼哈顿.png", prewitt_noisy_otsu_manhattan);

    // Sobel算子OTSU二值化
    cv::Mat sobel_otsu_euclidean, sobel_otsu_manhattan;
    cv::threshold(sobel_euclidean_8u, sobel_otsu_euclidean, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(sobel_manhattan_8u, sobel_otsu_manhattan, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imwrite("output/OTSU二值化/Sobel_OTSU二值化欧几里德.png", sobel_otsu_euclidean);
    cv::imwrite("output/OTSU二值化/Sobel_OTSU二值化曼哈顿.png", sobel_otsu_manhattan);

    // Sobel噪声图像OTSU二值化
    cv::Mat sobel_noisy_otsu_euclidean, sobel_noisy_otsu_manhattan;
    cv::threshold(sobel_noisy_euclidean_8u, sobel_noisy_otsu_euclidean, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(sobel_noisy_manhattan_8u, sobel_noisy_otsu_manhattan, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::imwrite("output/OTSU二值化/噪声Sobel_OTSU二值化欧几里德.png", sobel_noisy_otsu_euclidean);
    cv::imwrite("output/OTSU二值化/噪声Sobel_OTSU二值化曼哈顿.png", sobel_noisy_otsu_manhattan);

    // LoG算子OTSU二值化
    cv::Mat log_otsu, log_noisy_otsu, log_small_otsu, log_large_otsu;
    cv::threshold(log_result_8u, log_otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(log_noisy_result_8u, log_noisy_otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(log_small_sigma_8u, log_small_otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::threshold(log_large_sigma_8u, log_large_otsu, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);

    cv::imwrite("output/OTSU二值化/LoG_OTSU二值化.png", log_otsu);
    cv::imwrite("output/OTSU二值化/噪声LoG_OTSU二值化.png", log_noisy_otsu);
    cv::imwrite("output/OTSU二值化/LoG小sigma_OTSU二值化.png", log_small_otsu);
    cv::imwrite("output/OTSU二值化/LoG大sigma_OTSU二值化.png", log_large_otsu);

    // 生成OTSU二值化的白底黑线版本
    cv::Mat roberts_otsu_inverted_euc, roberts_otsu_inverted_man;
    cv::Mat prewitt_otsu_inverted_euc, prewitt_otsu_inverted_man;
    cv::Mat sobel_otsu_inverted_euc, sobel_otsu_inverted_man;
    cv::Mat log_otsu_inverted;

    cv::bitwise_not(roberts_otsu_euclidean, roberts_otsu_inverted_euc);
    cv::bitwise_not(roberts_otsu_manhattan, roberts_otsu_inverted_man);
    cv::bitwise_not(prewitt_otsu_euclidean, prewitt_otsu_inverted_euc);
    cv::bitwise_not(prewitt_otsu_manhattan, prewitt_otsu_inverted_man);
    cv::bitwise_not(sobel_otsu_euclidean, sobel_otsu_inverted_euc);
    cv::bitwise_not(sobel_otsu_manhattan, sobel_otsu_inverted_man);
    cv::bitwise_not(log_otsu, log_otsu_inverted);

    cv::imwrite("output/OTSU二值化/Roberts_OTSU白底黑线欧几里德.png", roberts_otsu_inverted_euc);
    cv::imwrite("output/OTSU二值化/Roberts_OTSU白底黑线曼哈顿.png", roberts_otsu_inverted_man);
    cv::imwrite("output/OTSU二值化/Prewitt_OTSU白底黑线欧几里德.png", prewitt_otsu_inverted_euc);
    cv::imwrite("output/OTSU二值化/Prewitt_OTSU白底黑线曼哈顿.png", prewitt_otsu_inverted_man);
    cv::imwrite("output/OTSU二值化/Sobel_OTSU白底黑线欧几里德.png", sobel_otsu_inverted_euc);
    cv::imwrite("output/OTSU二值化/Sobel_OTSU白底黑线曼哈顿.png", sobel_otsu_inverted_man);
    cv::imwrite("output/OTSU二值化/LoG_OTSU白底黑线.png", log_otsu_inverted);

    std::print("所有Roberts、Prewitt、Sobel和LoG算子图像处理完成并已保存！\n");
    std::print("\n生成的图像文件目录结构：\n");
    std::print("output/\n");
    std::print("├── 原始图像/\n");
    std::print("│   ├── 原始灰度图像.png\n");
    std::print("│   └── 噪声图像.png\n");
    std::print("├── Roberts算子/\n");
    std::print("│   ├── 边缘检测结果、梯度计算、二值化和白底黑线图像\n");
    std::print("├── Prewitt算子/\n");
    std::print("│   ├── 边缘检测结果、梯度计算、二值化和白底黑线图像\n");
    std::print("├── Sobel算子/\n");
    std::print("│   ├── 边缘检测结果、梯度计算、二值化和白底黑线图像\n");
    std::print("├── LoG算子/\n");
    std::print("│   ├── 边缘检测结果、不同参数测试、二值化和白底黑线图像\n");
    std::print("└── OTSU二值化/\n");
    std::print("    └── 所有算子的OTSU自适应二值化结果\n");
    std::print("\n每个目录包含原始图像和噪声图像的对应处理结果\n");
    std::print("OTSU方法优势：自适应确定最佳阈值，无需手动调参\n");

    return 0;
}
