#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * @file utils.h
 * @brief 图像处理工具函数集合
 * @details 实现完整的图像处理流程：图像增强 → 二值化 → 形态学处理 → 连通域分析 → 特征提取
 * 
 * 处理流程：
 * 1. 图像增强技术：灰度化、去噪、点运算（Log/幂变换）
 * 2. 二值化操作：OTSU阈值分割、形态学修复
 * 3. 连通域分析：提取、特征计算、统计分析
 */

/**
 * @brief 连通域特征结构体
 * @details 存储单个连通域的所有几何和灰度特征信息
 */
struct ConnectedComponentFeatures {
  double meanGray;        ///< 平均灰度值
  double minGray;         ///< 最小灰度值
  double maxGray;         ///< 最大灰度值
  cv::Moments moments;    ///< OpenCV几何矩结构（包含二阶矩和三阶矩）
  cv::Rect boundingRect;  ///< 最小外接矩形
  double area;            ///< 连通域面积（像素数）

  // 便于访问的几何矩getter方法
  constexpr double m20() const noexcept { return moments.m20; }  ///< 二阶矩 x^2
  constexpr double m02() const noexcept { return moments.m02; }  ///< 二阶矩 y^2
  constexpr double m11() const noexcept { return moments.m11; }  ///< 二阶矩 xy
  constexpr double m30() const noexcept { return moments.m30; }  ///< 三阶矩 x^3
  constexpr double m03() const noexcept { return moments.m03; }  ///< 三阶矩 y^3
  constexpr double m21() const noexcept { return moments.m21; }  ///< 三阶矩 x^2y
  constexpr double m12() const noexcept { return moments.m12; }  ///< 三阶矩 xy^2
};

// ==================== 第一部分：图像增强技术 ====================

/**
 * @brief 彩色图像灰度化处理
 * @details 将三通道彩色图像转换为单通道灰度图像，便于后续处理
 * @param colorImage 输入的彩色图像（BGR格式）
 * @return cv::Mat 转换后的灰度图像
 * @note 如果输入已经是单通道图像，则直接克隆返回
 */
cv::Mat convertToGrayscale(const cv::Mat& colorImage);

/**
 * @brief 低通滤波去噪处理
 * @details 使用高斯滤波器对图像进行去噪，去除高频噪声
 * @param image 输入的灰度图像
 * @return cv::Mat 滤波后的图像
 * @note 使用5x5高斯核，标准差为1.0
 */
cv::Mat applyLowPassFilter(const cv::Mat& image);

/**
 * @brief Log变换图像增强
 * @details 对数变换可以增强图像的暗部细节，压缩亮部动态范围
 * @param image 输入的灰度图像
 * @param c Log变换系数，控制变换强度，默认为1.0
 * @return cv::Mat 经过Log变换的图像
 * @note 变换公式：s = c * log(1 + r)，其中r为输入像素值
 */
cv::Mat applyLogTransform(const cv::Mat& image, double c = 1.0);

/**
 * @brief 幂变换（Gamma校正）
 * @details 幂变换用于调整图像的整体亮度和对比度
 * @param image 输入的灰度图像  
 * @param gamma 幂指数，控制亮度校正程度，默认为0.5
 * @return cv::Mat 经过幂变换的图像
 * @note 变换公式：s = r^gamma，gamma<1时提亮暗部，gamma>1时压暗亮部
 */
cv::Mat applyPowerTransform(const cv::Mat& image, double gamma = 0.5);

// ==================== 第二部分：二值化和形态学处理 ====================

/**
 * @brief OTSU自适应阈值二值化
 * @details 使用OTSU算法自动选择最佳阈值进行二值化，将字符变为白色，背景变为黑色
 * @param image 输入的灰度图像
 * @return cv::Mat 二值化后的图像（0或255）
 * @note OTSU算法通过最大化类间方差来自动确定最优阈值
 */
cv::Mat applyOTSUBinarization(const cv::Mat& image);

/**
 * @brief 形态学开运算去除噪点
 * @details 对二值化图像执行形态学开运算，去除小的噪声点和毛刺
 * @param binaryImage 输入的二值化图像
 * @param kernelSize 结构元素大小，默认为3
 * @return cv::Mat 形态学处理后的图像
 * @note 开运算 = 腐蚀 + 膨胀，可以去除小噪点但保持主要结构
 */
cv::Mat applyMorphologyOpening(const cv::Mat& binaryImage, int kernelSize = 3);

/**
 * @brief 形态学闭运算修复断开笔画
 * @details 对二值化图像执行形态学闭运算，连接断开的字符笔画
 * @param binaryImage 输入的二值化图像
 * @param kernelSize 结构元素大小，默认为3
 * @return cv::Mat 形态学处理后的图像
 * @note 闭运算 = 膨胀 + 腐蚀，可以填补小的空洞和连接断开的部分
 */
cv::Mat applyMorphologyClosing(const cv::Mat& binaryImage, int kernelSize = 3);

// ==================== 第三部分：连通域分析和特征提取 ====================

/**
 * @brief 连通域提取
 * @details 从二值化图像中提取所有外部轮廓作为连通域
 * @param binaryImage 输入的二值化图像
 * @return std::vector<std::vector<cv::Point>> 所有连通域的轮廓点集合
 * @note 使用cv::RETR_EXTERNAL模式只提取外部轮廓，避免嵌套轮廓
 */
std::vector<std::vector<cv::Point>> extractConnectedComponents(
    const cv::Mat& binaryImage);

/**
 * @brief 单个连通域特征提取
 * @details 计算单个连通域的完整特征信息，包括几何特征和灰度特征
 * @param contour 连通域轮廓点集
 * @param grayImage 原始灰度图像（用于计算灰度特征）
 * @return ConnectedComponentFeatures 连通域特征结构体
 * @note 使用ROI优化，只处理连通域的边界矩形区域以提高性能
 */
ConnectedComponentFeatures extractFeatures(
    const std::vector<cv::Point>& contour, const cv::Mat& grayImage);

/**
 * @brief 批量连通域特征提取
 * @details 对所有连通域批量进行特征提取
 * @param contours 所有连通域轮廓集合
 * @param grayImage 原始灰度图像
 * @return std::vector<ConnectedComponentFeatures> 所有连通域特征集合
 * @note 自动过滤空轮廓，使用emplace_back优化内存分配
 */
std::vector<ConnectedComponentFeatures> extractAllFeatures(
    const std::vector<std::vector<cv::Point>>& contours,
    const cv::Mat& grayImage);

// ==================== 第四部分：辅助工具函数 ====================

/**
 * @brief 创建输出目录结构
 * @details 为各个处理步骤创建相应的输出目录
 * @note 创建目录：grayscale, filtered, point_operations, binarization, morphology, contours
 */
void createOutputDirectories();

/**
 * @brief 连通域可视化
 * @details 将连通域绘制为彩色轮廓图，并添加编号和外接矩形
 * @param contours 连通域轮廓集合
 * @param features 连通域特征集合
 * @param imageSize 输出图像尺寸
 * @param outputPath 输出文件路径
 * @note 使用不同颜色区分各个连通域，绿色矩形表示外接矩形
 */
void visualizeConnectedComponents(
    const std::vector<std::vector<cv::Point>>& contours,
    const std::vector<ConnectedComponentFeatures>& features,
    const cv::Size& imageSize, const std::string& outputPath);

/**
 * @brief 打印连通域特征信息
 * @details 格式化输出所有连通域的详细特征统计信息
 * @param features 连通域特征集合
 * @note 输出包括：面积、外接矩形、灰度统计、二阶矩、三阶矩
 */
void printFeatureInfo(const std::vector<ConnectedComponentFeatures>& features);
