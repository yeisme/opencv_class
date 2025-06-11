#include <filesystem>
#include <opencv2/opencv.hpp>
#include <print>

/**
 * @brief 空间域高斯滤波
 * @param input 输入图像
 * @param ksize 高斯核大小（必须为奇数）
 * @param sigma 高斯核标准差
 * @return cv::Mat 滤波后的图像
 * @note 高斯滤波是常用的平滑滤波器，用于去除噪声和模糊图像
 */
cv::Mat spatialGaussianFilter(const cv::Mat &input, int ksize, double sigma) {
  cv::Mat result;
  cv::GaussianBlur(input, result, cv::Size(ksize, ksize), sigma);
  return result;
}

/**
 * @brief 空间域拉普拉斯滤波（用于高通滤波对比）
 * @param input 输入图像
 * @return cv::Mat 滤波后的图像
 * @note 拉普拉斯滤波器是二阶导数滤波器，用于边缘检测和图像锐化
 */
cv::Mat spatialLaplacianFilter(const cv::Mat &input) {
  cv::Mat result, temp;
  cv::Laplacian(input, temp, CV_16S, 3);
  cv::convertScaleAbs(temp, result);
  return result;
}

/**
 * @brief 创建理想低通滤波器
 * @param size 滤波器尺寸
 * @param radius 截止频率半径（以像素为单位）
 * @return cv::Mat 理想低通滤波器矩阵（CV_32F类型）
 * 
 * @details 理想低通滤波器工作原理：
 *          1. 频域中心点 (center.x, center.y) 代表零频率（直流分量）
 *          2. 距离中心点越远，频率越高
 *          3. 对每个频率点计算其到中心的欧几里得距离
 *          4. 距离 <= radius 的频率分量：完全通过（系数=1.0）
 *          5. 距离 > radius 的频率分量：完全阻止（系数=0.0）
 * 
 * @note 理想低通滤波器特点：
 *       - 优点：完美的频率选择性，截止频率内无衰减
 *       - 缺点：会产生振铃效应（Gibbs现象），因为频域中的尖锐截断
 *              对应空间域中的sinc函数，具有振荡特性
 *       - 应用：去除高频噪声，图像平滑，但实际中更常用高斯滤波器
 * 
 * @warning 振铃效应：在图像边缘附近会出现明显的波纹状伪影
 */
cv::Mat createIdealLowPassFilter(cv::Size size, double radius) {
  cv::Mat filter = cv::Mat::zeros(size, CV_32F);
  cv::Point center(size.width / 2, size.height / 2);  // 频域中心点（零频率点）

  for (int i = 0; i < size.height; i++) {
    for (int j = 0; j < size.width; j++) {
      double distance =
          std::sqrt(std::pow(i - center.y, 2) + std::pow(j - center.x, 2));

      filter.at<float>(i, j) = distance <= radius ? 1.0 : 0.0;
    }
  }
  return filter;
}

/**
 * @brief 创建理想高通滤波器
 * @param size 滤波器尺寸
 * @param radius 截止频率半径
 * @return cv::Mat 理想高通滤波器矩阵
 * @note 理想高通滤波器通过1减去理想低通滤波器得到
 *       用于保留高频分量，去除低频分量
 */
cv::Mat createIdealHighPassFilter(cv::Size size, double radius) {
  cv::Mat filter = createIdealLowPassFilter(size, radius);
  return 1.0 - filter;
}

/**
 * @brief 频域滤波函数
 * @param input 输入图像（单通道灰度图像）
 * @param filter 频域滤波器（与填充后图像尺寸相同的浮点矩阵）
 * @return cv::Mat 滤波后的图像
 * @note 该函数执行以下步骤：
 *       1. 对输入图像进行零填充以优化DFT计算
 *       2. 执行正向DFT变换到频域
 *       3. 进行频谱中心化
 *       4. 应用频域滤波器
 *       5. 进行频谱反中心化
 *       6. 执行反向DFT变换回空间域
 *       7. 裁剪并归一化结果
 */
cv::Mat frequencyDomainFilter(const cv::Mat &input, const cv::Mat &filter) {
  cv::Mat padded;
  int m = cv::getOptimalDFTSize(input.rows);
  int n = cv::getOptimalDFTSize(input.cols);
  cv::copyMakeBorder(input, padded, 0, m - input.rows, 0, n - input.cols,
                     cv::BORDER_CONSTANT, cv::Scalar::all(0));

  // 确保输入是浮点类型
  cv::Mat paddedFloat;
  padded.convertTo(paddedFloat, CV_32F);

  /**
   * 转换到频域的关键步骤解析：
   * 
   * 1. 创建复数表示：
   *    - DFT需要处理复数数据：f(x,y) = real + j*imag
   *    - 实际图像只有实部，虚部为0
   *    - OpenCV要求输入为双通道矩阵 [实部通道, 虚部通道]
   */
  
  // 步骤1：构建复数图像的两个通道
  cv::Mat planes[] = {
    paddedFloat,                                      // 实部：原始图像数据
    cv::Mat::zeros(paddedFloat.size(), CV_32F)       // 虚部：全零矩阵
  };
  
  // 步骤2：合并实部和虚部为双通道复数矩阵
  // 结果：complexI 为 CV_32FC2 类型 (32位浮点，双通道)
  // 每个像素包含 [real, imag] 两个分量
  cv::Mat complexI;
  cv::merge(planes, 2, complexI);
  
  /**
   * 步骤3：执行离散傅里叶变换 (DFT)
   * 
   * 数学原理：F(u,v) = ∑∑ f(x,y) * e^(-j2π(ux/M + vy/N))
   * 其中：
   * - f(x,y): 空间域图像函数
   * - F(u,v): 频域函数 (复数)
   * - (u,v): 频域坐标
   * - (x,y): 空间域坐标
   * - M,N: 图像尺寸
   * 
   * 物理意义：
   * - 将空间域的像素强度分解为不同频率的正弦/余弦分量
   * - 低频分量：图像的平滑区域、整体亮度
   * - 高频分量：图像的边缘、细节、纹理、噪声
   * 
   * 结果特点：
   * - 输出仍为CV_32FC2类型的复数矩阵
   * - 实部：余弦分量的幅值
   * - 虚部：正弦分量的幅值
   * - 幅值：sqrt(real² + imag²) 表示该频率的强度
   * - 相位：atan2(imag, real) 表示该频率的相位信息
   */
  cv::dft(complexI, complexI);

  // 中心化频谱 - 同时处理实部和虚部
  cv::split(complexI, planes);

  // 对实部进行中心化
  int cx = planes[0].cols / 2;
  int cy = planes[0].rows / 2;

  for (int i = 0; i < 2; i++) {  // 处理实部和虚部
    cv::Mat q0(planes[i], cv::Rect(0, 0, cx, cy));
    cv::Mat q1(planes[i], cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(planes[i], cv::Rect(0, cy, cx, cy));
    cv::Mat q3(planes[i], cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
  }

  // 重新合并
  cv::merge(planes, 2, complexI);

  // 确保滤波器尺寸与填充后的图像一致
  cv::Mat resizedFilter;
  if (filter.size() != paddedFloat.size()) {
    cv::resize(filter, resizedFilter, paddedFloat.size());
  } else {
    resizedFilter = filter;
  }

  // 应用滤波器 - 创建复数滤波器
  cv::Mat filterPlanes[] = {resizedFilter,
                            cv::Mat::zeros(resizedFilter.size(), CV_32F)};
  cv::Mat filterComplex;
  cv::merge(filterPlanes, 2, filterComplex);

  // 应用滤波器
  cv::mulSpectrums(complexI, filterComplex, complexI, 0);

  // 反中心化
  cv::split(complexI, planes);
  for (int i = 0; i < 2; i++) {
    cv::Mat q0(planes[i], cv::Rect(0, 0, cx, cy));
    cv::Mat q1(planes[i], cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(planes[i], cv::Rect(0, cy, cx, cy));
    cv::Mat q3(planes[i], cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
  }
  cv::merge(planes, 2, complexI);

  // 反变换
  cv::idft(complexI, complexI);
  cv::split(complexI, planes);
  cv::magnitude(planes[0], planes[1], planes[0]);

  // 裁剪回原始尺寸
  cv::Mat result = planes[0](cv::Rect(0, 0, input.cols, input.rows));

  // 归一化并转换为CV_8U
  cv::Mat resultNormalized;
  cv::normalize(result, resultNormalized, 0, 255, cv::NORM_MINMAX);
  resultNormalized.convertTo(resultNormalized, CV_8U);

  return resultNormalized;
}

/**
 * @brief 主函数 - 演示空间域和频域滤波的对比
 * @return int 程序退出状态码
 * @note 该程序对房间图像和数字图像分别进行：
 *       - 频域理想低通/高通滤波
 *       - 空间域高斯/拉普拉斯滤波
 *       并将结果保存到不同文件夹中进行对比分析
 */
int main() {
  // 创建输出文件夹
  std::filesystem::create_directories("输出结果/原始图像");
  std::filesystem::create_directories("输出结果/低通滤波");
  std::filesystem::create_directories("输出结果/高通滤波");

  // 1. 加载图像
  cv::Mat roomImage = cv::imread("img/room.tif", cv::IMREAD_GRAYSCALE);
  cv::Mat numberImage = cv::imread("img/number.tif", cv::IMREAD_GRAYSCALE);

  if (roomImage.empty() || numberImage.empty()) {
    std::println("无法加载图像，请确保图像文件存在");
    return -1;
  }

  // 保存原始图像
  cv::imwrite("输出结果/原始图像/房间_原图.jpg", roomImage);
  cv::imwrite("输出结果/原始图像/数字_原图.jpg", numberImage);

  // 2. 低通滤波处理
  double lowPassRadius = 30;
  int gaussianKsize = 15;
  double gaussianSigma = 5;

  // 创建与填充后尺寸匹配的滤波器
  int m_room = cv::getOptimalDFTSize(roomImage.rows);
  int n_room = cv::getOptimalDFTSize(roomImage.cols);
  int m_number = cv::getOptimalDFTSize(numberImage.rows);
  int n_number = cv::getOptimalDFTSize(numberImage.cols);

  // 频域低通滤波
  cv::Mat roomLowPassFreq = frequencyDomainFilter(
      roomImage,
      createIdealLowPassFilter(cv::Size(n_room, m_room), lowPassRadius));
  cv::Mat numberLowPassFreq = frequencyDomainFilter(
      numberImage,
      createIdealLowPassFilter(cv::Size(n_number, m_number), lowPassRadius));

  // 空间域高斯滤波
  cv::Mat roomLowPassSpatial =
      spatialGaussianFilter(roomImage, gaussianKsize, gaussianSigma);
  cv::Mat numberLowPassSpatial =
      spatialGaussianFilter(numberImage, gaussianKsize, gaussianSigma);

  // 保存低通滤波结果
  cv::imwrite("输出结果/低通滤波/房间_频域低通滤波.jpg", roomLowPassFreq);
  cv::imwrite("输出结果/低通滤波/房间_空间域高斯滤波.jpg", roomLowPassSpatial);
  cv::imwrite("输出结果/低通滤波/数字_频域低通滤波.jpg", numberLowPassFreq);
  cv::imwrite("输出结果/低通滤波/数字_空间域高斯滤波.jpg",
              numberLowPassSpatial);

  // 3. 高通滤波处理
  double highPassRadius = 30;

  // 频域高通滤波
  cv::Mat roomHighPassFreq = frequencyDomainFilter(
      roomImage,
      createIdealHighPassFilter(cv::Size(n_room, m_room), highPassRadius));
  cv::Mat numberHighPassFreq = frequencyDomainFilter(
      numberImage,
      createIdealHighPassFilter(cv::Size(n_number, m_number), highPassRadius));

  // 空间域拉普拉斯滤波
  cv::Mat roomHighPassSpatial = spatialLaplacianFilter(roomImage);
  cv::Mat numberHighPassSpatial = spatialLaplacianFilter(numberImage);

  // 保存高通滤波结果
  cv::imwrite("输出结果/高通滤波/房间_频域高通滤波.jpg", roomHighPassFreq);
  cv::imwrite("输出结果/高通滤波/房间_空间域拉普拉斯滤波.jpg",
              roomHighPassSpatial);
  cv::imwrite("输出结果/高通滤波/数字_频域高通滤波.jpg", numberHighPassFreq);
  cv::imwrite("输出结果/高通滤波/数字_空间域拉普拉斯滤波.jpg",
              numberHighPassSpatial);

  std::println("所有图像处理完成，结果已保存到'输出结果'文件夹中");
  return 0;
}