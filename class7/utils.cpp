#include "utils.h"

#include <algorithm>
#include <filesystem>
#include <print>

// ==================== 第一部分：图像增强技术实现 ====================

// 灰度化处理 - 将三通道彩色图像转换为单通道灰度图像
cv::Mat convertToGrayscale(const cv::Mat& colorImage) {
  cv::Mat grayImage;

  // 检查输入图像通道数，决定是否需要转换
  if (colorImage.channels() == 3) {
    // BGR彩色图像转灰度：使用加权平均 Gray = 0.299*R + 0.587*G + 0.114*B
    cv::cvtColor(colorImage, grayImage, cv::COLOR_BGR2GRAY);
  } else {
    // 已经是单通道图像，直接克隆
    grayImage = colorImage.clone();
  }
  return grayImage;
}

// 低通滤波去噪 - 使用高斯滤波器
cv::Mat applyLowPassFilter(const cv::Mat& image) {
  cv::Mat filteredImage;

  // 高斯滤波去噪：
  // - 核大小5x5：足够去除细小噪声，不会过度模糊
  // - 标准差1.0：适中的平滑程度
  cv::GaussianBlur(image, filteredImage, cv::Size(5, 5), 1.0);
  return filteredImage;
}

// Log变换 - 增强暗部细节
cv::Mat applyLogTransform(const cv::Mat& image, double c) {
  cv::Mat result;

  // 转换为双精度浮点型以避免数值溢出
  image.convertTo(result, CV_64F);

  // 应用Log变换: s = c * log(1 + r)
  // +1是为了避免log(0)的数学错误
  cv::log(result + 1.0, result);
  result *= c;  // 乘以系数c控制变换强度

  // 将结果归一化到[0,255]范围并转回8位整数
  cv::normalize(result, result, 0, 255, cv::NORM_MINMAX);
  result.convertTo(result, CV_8U);

  return result;
}

// 幂变换（Gamma校正）- 调整图像亮度
cv::Mat applyPowerTransform(const cv::Mat& image, double gamma) {
  cv::Mat result;

  // 转换为双精度以保证计算精度
  image.convertTo(result, CV_64F);

  // 归一化到[0,1]范围进行幂运算
  result /= 255.0;

  // 应用幂变换: s = r^gamma
  // gamma < 1: 提亮暗部，增强暗部细节
  // gamma > 1: 压暗亮部，突出亮部信息
  cv::pow(result, gamma, result);

  // 转换回[0,255]范围
  result *= 255.0;
  result.convertTo(result, CV_8U);

  return result;
}

// ==================== 第二部分：二值化和形态学处理实现 ====================

// OTSU二值化 - 自动选择最佳阈值进行二值化
cv::Mat applyOTSUBinarization(const cv::Mat& image) {
  cv::Mat binaryImage;
  cv::Mat grayImage;

  // 确保输入是灰度图像
  if (image.channels() == 3) {
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
  } else {
    grayImage = image.clone();
  }

  // OTSU二值化：
  // - THRESH_BINARY: 大于阈值的像素设为maxval(255)，小于阈值的设为0
  // - THRESH_OTSU: 自动计算最优阈值，使类间方差最大
  // 结果：字符(前景)为白色255，背景为黑色0
  cv::threshold(grayImage, binaryImage, 0, 255,
                cv::THRESH_BINARY + cv::THRESH_OTSU);

  return binaryImage;
}

// 形态学开运算 - 去除小噪点
cv::Mat applyMorphologyOpening(const cv::Mat& binaryImage, int kernelSize) {
  cv::Mat result;

  // 创建椭圆形结构元素
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             cv::Size(kernelSize, kernelSize));

  // 形态学开运算 = 先腐蚀后膨胀：
  // 1. 腐蚀：去除小的噪声点和细小连接
  // 2. 膨胀：恢复主要结构的大小
  cv::morphologyEx(binaryImage, result, cv::MORPH_OPEN, kernel);

  return result;
}

// 形态学闭运算 - 修复断开的字符笔画
cv::Mat applyMorphologyClosing(const cv::Mat& binaryImage, int kernelSize) {
  cv::Mat result;

  // 创建椭圆形结构元素：
  // - 椭圆形比矩形更自然，能更好地保持字符形状
  // - kernelSize控制操作强度：越大连接能力越强，但可能连接不该连接的部分
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                             cv::Size(kernelSize, kernelSize));

  // 形态学闭运算 = 先膨胀后腐蚀：
  // 1. 膨胀：扩大前景区域，连接断开的笔画
  // 2. 腐蚀：恢复原有大小，但保持连接效果
  cv::morphologyEx(binaryImage, result, cv::MORPH_CLOSE, kernel);

  return result;
}

// ==================== 第三部分：连通域分析和特征提取实现 ====================

// 连通域提取 - 获取各个连通区域的轮廓
std::vector<std::vector<cv::Point>> extractConnectedComponents(
    const cv::Mat& binaryImage) {
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Vec4i> hierarchy;

  // 轮廓检测：
  // - RETR_EXTERNAL: 只检测最外层轮廓，忽略内部孔洞
  // - CHAIN_APPROX_SIMPLE: 压缩轮廓，只保存端点，节省内存
  cv::findContours(binaryImage, contours, hierarchy, cv::RETR_EXTERNAL,
                   cv::CHAIN_APPROX_SIMPLE);

  return contours;
}

// 单个连通域特征提取
ConnectedComponentFeatures extractFeatures(
    const std::vector<cv::Point>& contour, const cv::Mat& grayImage) {
  // 输入验证：确保轮廓和图像都不为空
  if (contour.empty() || grayImage.empty()) {
    return {};  // 返回默认构造的特征对象
  }

  ConnectedComponentFeatures features;

  // 1. 计算基本几何特征
  features.boundingRect = cv::boundingRect(contour);  // 最小外接矩形
  features.area = cv::contourArea(contour);           // 连通域面积
  features.moments = cv::moments(contour);  // 几何矩（包含二阶矩和三阶矩）

  // 2. ROI优化：只处理连通域的边界矩形区域
  // 这样可以显著减少内存使用和计算量
  const cv::Rect& roi = features.boundingRect;
  cv::Mat roiMask = cv::Mat::zeros(roi.size(), CV_8UC1);

  // 将轮廓坐标转换到ROI坐标系
  std::vector<cv::Point> adjustedContour;
  adjustedContour.reserve(contour.size());
  std::transform(contour.begin(), contour.end(),
                 std::back_inserter(adjustedContour),
                 [&roi](const cv::Point& p) {
                   return cv::Point(p.x - roi.x, p.y - roi.y);
                 });

  // 在ROI掩码中填充连通域区域
  cv::fillPoly(roiMask, std::vector<std::vector<cv::Point>>{adjustedContour},
               cv::Scalar(255));

  // 3. 提取ROI区域的灰度图像
  cv::Mat grayROI = grayImage(roi);

  // 4. 计算灰度统计特征
  cv::Scalar meanScalar;
  // 只计算掩码区域内的统计信息
  cv::meanStdDev(grayROI, meanScalar, cv::noArray(), roiMask);
  features.meanGray = meanScalar[0];  // 平均灰度

  // 5. 找到连通域内的灰度极值
  double minVal, maxVal;
  cv::minMaxLoc(grayROI, &minVal, &maxVal, nullptr, nullptr, roiMask);
  features.minGray = minVal;  // 最小灰度值
  features.maxGray = maxVal;  // 最大灰度值

  return features;
}

// 批量提取所有连通域特征 - 优化版本
std::vector<ConnectedComponentFeatures> extractAllFeatures(
    const std::vector<std::vector<cv::Point>>& contours,
    const cv::Mat& grayImage) {
  // 输入验证
  if (contours.empty() || grayImage.empty()) {
    return {};
  }

  // 预分配内存以提高性能
  std::vector<ConnectedComponentFeatures> allFeatures;
  allFeatures.reserve(contours.size());

  // 批量处理所有连通域
  for (const auto& contour : contours) {
    if (!contour.empty()) {  // 过滤空轮廓
      // 使用emplace_back直接在容器中构造对象，避免拷贝
      allFeatures.emplace_back(extractFeatures(contour, grayImage));
    }
  }

  return allFeatures;
}

// ==================== 第四部分：辅助工具函数实现 ====================

namespace fs = std::filesystem;

void createOutputDirectories() {
  // 定义所有需要创建的目录
  const std::vector<std::string> dirs = {
      "output/grayscale",         // 灰度化结果
      "output/filtered",          // 滤波结果
      "output/point_operations",  // 点运算结果
      "output/binarization",      // 二值化结果
      "output/morphology",        // 形态学处理结果
      "output/contours"           // 连通域可视化结果
  };

  // 批量创建目录（包括父目录）
  for (const auto& dir : dirs) {
    fs::create_directories(dir);
  }
}

void visualizeConnectedComponents(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<ConnectedComponentFeatures>& features,
    const cv::Size& imageSize, const std::string& outputPath) {
  // 创建黑色背景的彩色图像用于可视化
  cv::Mat contoursImage = cv::Mat::zeros(imageSize, CV_8UC3);

  // 为每个连通域绘制可视化信息
  for (size_t i = 0; i < contours.size() && i < features.size(); ++i) {
    // 使用伪随机颜色生成器，确保相邻连通域颜色差异明显
    cv::Scalar color =
        cv::Scalar((i * 67) % 256, (i * 137) % 256, (i * 211) % 256);

    // 绘制连通域轮廓（彩色）
    cv::drawContours(contoursImage, contours, static_cast<int>(i), color, 2);

    // 绘制最小外接矩形（绿色）
    cv::rectangle(contoursImage, features[i].boundingRect,
                  cv::Scalar(0, 255, 0), 2);

    // 添加连通域编号标签
    cv::Point textPos(features[i].boundingRect.x,
                      features[i].boundingRect.y - 5);
    cv::putText(contoursImage, std::to_string(i + 1), textPos,
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
  }

  // 保存可视化结果
  cv::imwrite(outputPath, contoursImage);
}

void printFeatureInfo(const std::vector<ConnectedComponentFeatures>& features) {
  std::print("\n===== 连通域特征统计 =====\n");

  // 逐个输出每个连通域的详细特征信息
  for (size_t i = 0; i < features.size(); ++i) {
    const auto& feat = features[i];

    std::print("\n连通域 {}:\n", i + 1);

    // 基本几何特征
    std::print("  面积: {:.2f} 像素\n", feat.area);
    std::print("  外接矩形: ({}, {}) {}×{}\n", feat.boundingRect.x,
               feat.boundingRect.y, feat.boundingRect.width,
               feat.boundingRect.height);

    // 灰度统计特征
    std::print("  灰度统计: 均值={:.2f}, 范围=[{:.0f}, {:.0f}]\n",
               feat.meanGray, feat.minGray, feat.maxGray);

    // 二阶矩（描述形状的分布特征）
    std::print("  二阶矩: m20={:.2f}, m02={:.2f}, m11={:.2f}\n", feat.m20(),
               feat.m02(), feat.m11());

    // 三阶矩（描述形状的不对称性）
    std::print("  三阶矩: m30={:.2f}, m03={:.2f}, m21={:.2f}, m12={:.2f}\n",
               feat.m30(), feat.m03(), feat.m21(), feat.m12());
  }
}