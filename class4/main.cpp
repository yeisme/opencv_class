#include <opencv2/opencv.hpp>
#include <print>
#include <string>
#include <vector>
#include <cmath>    // For std::sqrt, std::abs if needed, though cv:: versions are used
#include <algorithm> // For std::clamp

// 从 class3/main.cpp 借鉴并稍作调整以适配
/**
 * @brief 为灰度图像添加高斯噪声
 *
 * @param src 输入图像 (灰度图)
 * @param dst 输出图像
 * @param mean 高斯噪声的均值 (默认为0)
 * @param stddev 高斯噪声的标准差，控制噪声强度 (默认为15)
 */
void add_gaussian_noise(const cv::Mat& src, cv::Mat& dst, double mean = 0.0, double stddev = 15.0)
{
	CV_Assert(!src.empty() && src.channels() == 1); // 确保输入是单通道图像
	cv::Mat noise = cv::Mat(src.size(), CV_16SC1); // 使用16位有符号类型存储噪声以避免截断
	cv::randn(noise, mean, stddev);

	dst = cv::Mat::zeros(src.size(), src.type());
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			// 将噪声加到原始像素值上
			int value = static_cast<int>(src.at<uchar>(i, j)) + static_cast<short>(noise.at<short>(i, j));
			dst.at<uchar>(i, j) = static_cast<uchar>(std::clamp(value, 0, 255));
		}
	}
}

/**
 * @brief 使用Roberts算子执行边缘检测及相关处理
 * @param input_image 输入的单通道灰度图像
 * @param window_suffix 附加到OpenCV窗口标题的后缀，用于区分原始图像和噪声图像的处理结果
 */
void perform_roberts_edge_detection(const cv::Mat& input_image, const std::string& window_suffix)
{
	CV_Assert(!input_image.empty() && input_image.channels() == 1);

	// 定义Roberts算子
	// rh (用户定义的“水平”算子)
	cv::Mat kernel_rh = (cv::Mat_<float>(2, 2) << 0, 1, -1, 0);
	// rv (用户定义的“垂直”算子)
	cv::Mat kernel_rv = (cv::Mat_<float>(2, 2) << 1, 0, 0, -1);

	cv::Mat grad_rh_raw, grad_rv_raw;
	// 使用filter2D应用算子，输出类型为CV_32F以保留符号和精度
	cv::filter2D(input_image, grad_rh_raw, CV_32F, kernel_rh);
	cv::filter2D(input_image, grad_rv_raw, CV_32F, kernel_rv);

	// 显示处理后的“水平”和“垂直”边界检测结果
	cv::Mat display_grad_rh, display_grad_rv;
	cv::convertScaleAbs(grad_rh_raw, display_grad_rh); // 转换到CV_8U以便显示
	cv::convertScaleAbs(grad_rv_raw, display_grad_rv);
	cv::imshow("Roberts RH Edges" + window_suffix, display_grad_rh);
	cv::imshow("Roberts RV Edges" + window_suffix, display_grad_rv);

	// 计算梯度模 - 欧几里德距离: G = sqrt(grad_rh_raw^2 + grad_rv_raw^2)
	cv::Mat grad_rh_sq, grad_rv_sq, sum_sq, grad_euclidean_raw;
	cv::pow(grad_rh_raw, 2.0, grad_rh_sq);
	cv::pow(grad_rv_raw, 2.0, grad_rv_sq);
	cv::add(grad_rh_sq, grad_rv_sq, sum_sq);
	cv::sqrt(sum_sq, grad_euclidean_raw); // grad_euclidean_raw 是 CV_32F

	cv::Mat grad_euclidean_display;
	cv::normalize(grad_euclidean_raw, grad_euclidean_display, 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::imshow("Gradient Magnitude (Euclidean)" + window_suffix, grad_euclidean_display);

	// 计算梯度模 - 街区距离: G = |grad_rh_raw| + |grad_rv_raw|
	cv::Mat abs_grad_rh, abs_grad_rv, grad_manhattan_raw;
	abs_grad_rh = cv::abs(grad_rh_raw); // cv::abs 保留 CV_32F 类型
	abs_grad_rv = cv::abs(grad_rv_raw);
	cv::add(abs_grad_rh, abs_grad_rv, grad_manhattan_raw); // grad_manhattan_raw 是 CV_32F

	cv::Mat grad_manhattan_display;
	cv::normalize(grad_manhattan_raw, grad_manhattan_display, 0, 255, cv::NORM_MINMAX, CV_8U);
	cv::imshow("Gradient Magnitude (Manhattan)" + window_suffix, grad_manhattan_display);

	// 对欧几里德梯度模进行二值化处理
	// 提示：先做检测结果的直方图，参考直方图中灰度的分布尝试确定阈值；
	// 应反复调节阈值的大小，直至二值化的效果最为满意为止。
	// 这里使用一个占位符阈值。
	double threshold_value = 50.0; // 这是一个经验值，您应该根据图像直方图调整
	cv::Mat binarized_gradient;
	std::print("提示: 对于图像组 '{}', 二值化阈值当前为 {}. 请考虑分析 '{}' 的直方图以获得最佳阈值。\n",
		window_suffix, threshold_value, "Gradient Magnitude (Euclidean)" + window_suffix);
	cv::threshold(grad_euclidean_display, binarized_gradient, threshold_value, 255, cv::THRESH_BINARY);
	cv::imshow("Binarized Gradient (Euclidean)" + window_suffix, binarized_gradient);

	// 将处理结果转化为“白底黑线条”的方式
	cv::Mat binarized_inverted;
	cv::bitwise_not(binarized_gradient, binarized_inverted);
	cv::imshow("Binarized (White BG, Black Lines)" + window_suffix, binarized_inverted);
}

int main() {
	std::string image_path = "img/img1.png"; // 示例路径，请修改
	cv::Mat original_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

	if (original_image.empty()) {
		std::print("错误: 无法加载图像 '{}'. 请检查路径是否正确。\n", image_path);
		return -1;
	}

	cv::imshow("Original Grayscale Image", original_image);
	std::print("处理原始图像...\n");
	perform_roberts_edge_detection(original_image, " (Original)");

	// 给图像加上零均值的高斯噪声
	cv::Mat noisy_image;
	double noise_mean = 0.0;
	double noise_stddev = 20.0; // 可以调整噪声强度
	add_gaussian_noise(original_image, noisy_image, noise_mean, noise_stddev);
	cv::imshow("Noisy Image", noisy_image);

	std::print("\n处理带噪声的图像 (均值={}, 标准差={})...\n", noise_mean, noise_stddev);
	perform_roberts_edge_detection(noisy_image, " (Noisy)");

	cv::waitKey(0);
	return 0;
}
