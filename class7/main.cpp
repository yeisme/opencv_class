#include <chrono>
#include <opencv2/opencv.hpp>
#include <print>

#include "utils.h"

/**
 * @brief 主函数 - 完整的图像处理流水线
 * @details 实现三大处理阶段：
 *
 * 【第一阶段】图像增强技术：
 * - 彩色图像灰度化（三通道→单通道）
 * - 低通滤波去噪（高斯滤波）
 * - 点运算增强（Log变换和幂变换）
 *
 * 【第二阶段】二值化和形态学处理：
 * - OTSU自适应阈值二值化
 * - 形态学闭运算修复断开笔画
 * - 连通域提取
 *
 * 【第三阶段】特征分析：
 * - 连通域特征提取（几何+灰度特征）
 * - 结果可视化和统计输出
 */
int main() {
  // 记录程序开始时间，用于性能分析
  auto start = std::chrono::high_resolution_clock::now();

  // ==================== 输入图像加载 ====================
  auto colorImage = cv::imread("img/img1.png");

  if (colorImage.empty()) {
    std::print("错误: 无法加载图像文件 'img/img1.png'\n");
    std::print("请确保图像文件存在于程序运行目录下\n");
    return -1;
  }

  std::print("开始处理图像 ({}×{})...\n", colorImage.cols, colorImage.rows);

  // 创建输出目录结构
  createOutputDirectories();

  // ==================== 第一阶段：图像增强技术 ====================
  std::print("\n=== 第一阶段：图像增强技术 ===\n");

  // 1. 彩色图像灰度化处理（三通道→单通道）
  std::print("1. 执行灰度化处理...\n");
  cv::Mat grayImage = convertToGrayscale(colorImage);
  cv::imwrite("output/grayscale/gray_image.jpg", grayImage);

  // 2. 低通滤波去噪（高斯滤波）
  std::print("2. 执行低通滤波去噪...\n");
  cv::Mat filteredImage = applyLowPassFilter(grayImage);
  cv::imwrite("output/filtered/filtered_image.jpg", filteredImage);

  // 3. 点运算处理（Log变换和幂变换）
  std::print("3. 执行点运算增强...\n");
  cv::Mat logTransformed = applyLogTransform(grayImage, 15);  // Log变换增强暗部
  cv::Mat powerTransformed = applyPowerTransform(grayImage, 1.5);  // 幂变换提亮

  cv::imwrite("output/point_operations/log_transformed.jpg", logTransformed);
  cv::imwrite("output/point_operations/power_transformed.jpg",
              powerTransformed);

  // ==================== 第二阶段：二值化和形态学处理 ====================
  std::print("\n=== 第二阶段：二值化和形态学处理 ===\n");

  // 4. OTSU自适应阈值二值化（使用幂变换的结果）
  std::print("4. 执行OTSU二值化...\n");
  cv::Mat binaryImage = applyOTSUBinarization(powerTransformed);
  cv::imwrite("output/binarization/binary_image.jpg", binaryImage);

  // 5. 形态学修复处理
  std::print("5. 执行形态学修复...\n");

  // 先执行开运算去除小噪点
  std::print("   执行开运算去除噪点...\n");
  cv::Mat openingResult = applyMorphologyOpening(binaryImage, 2);  // 小核去噪
  cv::imwrite("output/morphology/morphology_opening.jpg", openingResult);

  // 再执行闭运算连接断开的笔画，使用更大的核
  std::print("   执行闭运算连接笔画...\n");
  cv::Mat closingResult =
      applyMorphologyClosing(openingResult, 2);
  cv::imwrite("output/morphology/morphology_closing.jpg", closingResult);

  // 6. 连通域提取
  std::print("6. 提取连通域...\n");
  std::vector<std::vector<cv::Point>> contours =
    extractConnectedComponents(closingResult);  // 使用最终形态学结果

  // 过滤面积过小的噪声连通域 - 调整过滤阈值
  std::print("   过滤前连通域数量: {}\n", contours.size());

  // 计算图像总面积的0.1%作为动态阈值
  double imageArea = closingResult.rows * closingResult.cols;
  double minArea =
      std::max(50.0, imageArea * 0.001);  // 最小50像素或图像面积的0.1%

  std::print("   使用动态过滤阈值: {:.1f} 像素\n", minArea);

  contours.erase(
      std::remove_if(contours.begin(), contours.end(),
                     [minArea](const std::vector<cv::Point>& contour) {
                       return cv::contourArea(contour) < minArea;
                     }),
      contours.end());

  if (contours.empty()) {
    std::print("警告: 未检测到有效的连通域\n");
    std::print("建议: 1) 检查二值化效果 2) 降低过滤阈值 3) 调整形态学参数\n");
    return 0;
  }

  std::print("   过滤后有效连通域数量: {}\n", contours.size());

  // 输出连通域面积统计
  std::vector<double> areas;
  for (const auto& contour : contours) {
    areas.push_back(cv::contourArea(contour));
  }
  std::sort(areas.begin(), areas.end());

  std::print("   连通域面积范围: {:.1f} - {:.1f} 像素\n", areas.front(),
             areas.back());
  if (areas.size() > 1) {
    std::print("   连通域面积中位数: {:.1f} 像素\n", areas[areas.size() / 2]);
  }

  // ==================== 第三阶段：特征分析 ====================
  std::print("\n=== 第三阶段：连通域特征分析 ===\n");

  // 7. 连通域特征提取
  std::print("7. 提取连通域特征...\n");
  std::vector<ConnectedComponentFeatures> features =
      extractAllFeatures(contours, grayImage);

  // 8. 结果可视化
  std::print("8. 生成可视化结果...\n");
  visualizeConnectedComponents(contours, features,
                               closingResult.size(),  // 使用最终结果
                               "output/contours/connected_components.jpg");

  // ==================== 处理结果统计和输出 ====================

  // 计算处理耗时
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  // 输出处理总结
  std::print("\n=== 处理完成总结 ===\n");
  std::print("总处理时间: {}ms\n", duration.count());
  std::print("最终检测到 {} 个有效连通域\n", contours.size());

  // 输出详细的连通域特征信息
  printFeatureInfo(features);

  // 保存原始图像到输出目录作为对比
  cv::imwrite("output/original_image.jpg", colorImage);

  std::print("\n所有结果已保存到 output/ 目录下\n");

  return 0;
}
