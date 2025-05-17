#include <iostream>
#include <cmath>
#include <exception>
#include <string>
#include <opencv2/opencv.hpp>
#include <format>
#include <immintrin.h>
#include <print>
#include <omp.h>


// 灰度取反函数
void invert_grayscale(cv::Mat& image, cv::Mat& output)
{
	// 检查图像是否为空或非灰度图
	CV_Assert(!image.empty() && image.channels() == 1);


	// 创建输出图像
	output = cv::Mat::zeros(image.size(), image.type());

	// 使用公式: s = (L-1) - r 进行灰度取反
	// 对于8位图像，L = 256，所以公式简化为: s = 255 - r
	output = 255 - image;

	// 保存为PNG格式
	CV_Assert(cv::imwrite("img/inverted_image.png", output));
}


// 对数变换
void log_transform(cv::Mat& image, cv::Mat& output)
{
	// 检查图像是否为空或非灰度图
	CV_Assert(!image.empty() && image.channels() == 1);

	// 创建输出图像
	output = cv::Mat::zeros(image.size(), image.type());
	// 对数变换公式: s = c * log(1 + r)
	const double c = 255.0 / std::log(256.0);
	for (int y = 0; y < image.rows; ++y)
	{
		for (int x = 0; x < image.cols; ++x)
		{
			output.at<uchar>(y, x) = static_cast<uchar>(c * std::log(1 + image.at<uchar>(y, x)));
		}
	}
	// 保存为PNG格式
	CV_Assert(cv::imwrite("img/log_transformed_image.png", output));
}


/**
 * @brief 图像的幂变换，无 IO，基础版本
 * @param image 输入图像 cv::Mat
 * @param output 输出图像 cv::Mat
 * @param c 幂变换的常数
 * @param gamma 幂变换的指数
 */
void power_transform(cv::Mat& image, cv::Mat& output, float c, float gamma)
{
	// 输入验证
	CV_Assert(!image.empty() && image.channels() == 1);
	CV_Assert(gamma > 0 && c > 0);

	// 特殊情况处理: gamma == 1
	if (std::abs(gamma - 1.0f) < 1e-6f)
	{
		output *= c;
		return;
	}

	// 转换为浮点型并归一化
	image.convertTo(output, CV_32F, 1.0 / 255.0);

	cv::pow(output, gamma, output);
	output *= c;

	// 反归一化并限制值域
	output *= 255.0;
	output = cv::min(cv::max(output, 0.0f), 255.0f);

	// 转换回8位无符号整型
	output.convertTo(output, CV_8U);
}

/**
 * @brief 图像的幂变换，保存图像
 */
void power_transform_with_save(cv::Mat& image, cv::Mat& output, float c, float gamma)
{
	power_transform(image, output, c, gamma);

	auto save_path = std::format("img/power_transformed_{:.1f}_{:.1f}.png", c, gamma);
	CV_Assert(cv::imwrite(save_path, output));
}

/**
 * @brief 图像的幂变换，无 IO，优化版本，这里需要说明：
 * 启用 OpenMP 并行化，使用 SIMD 指令集加速计算，使用 Lookup Table
 * 基于我的 CPU: 11th Gen Intel(R) Core(TM) i5-11260H (12) @ 4.40 GHz
 * 在 Intel 官方查看：https://www.intel.cn/content/www/cn/zh/products/sku/213806/intel-core-i511260h-processor-12m-cache-up-to-4-40-ghz/specifications.html
 * 发现支持以下拓展指令 Intel® SSE4.1, Intel® SSE4.2, Intel® AVX2, Intel® AVX-512
 * 再基于我的编译器 MSVC 19.40.33820, 可以使用 avx512 进行加速, 在 cmake 中使用 /O2 /openmp /arch:AVX512 启用
 * @param image 输入图像 cv::Mat
 * @param output 输出图像 cv::Mat
 * @param c 幂变换的常数
 * @param gamma 幂变换的指数
 */
void power_transform_better(cv::Mat& image, cv::Mat& output, float c, float gamma)
{
	// 输入验证
	CV_Assert(!image.empty() && image.channels() == 1);
	CV_Assert(gamma > 0 && c > 0);

	// 特殊情况处理: gamma ≈ 1
	if (std::abs(gamma - 1.0f) < 1e-6f)
	{
		output *= c;
		return;
	}

	// 计算 lookup Table
	cv::Mat lookUpTable(1, 256, CV_8U);
	uchar* lut = lookUpTable.ptr();
	for (int i = 0; i < 256; ++i)
	{
		auto val = c * std::pow(static_cast<float>(i) / 255.0f, gamma) * 255.0f;
		if (val > 255.0f)
			lut[i] = 255;
		else if (val < 0.0f)
			lut[i] = 0;
		else
			lut[i] = static_cast<uchar>(val + 0.5f);
	}

	// 确保输出图像是8位无符号整型
	output.create(image.size(), CV_8UC1);

	// 应用 lookup Table
	// 使用 OpenMP 并行化
#pragma omp parallel for
	for (int i = 0; i < image.rows; i++)
	{
		for (int j = 0; j < image.cols; j++)
		{
			uchar input_value = image.at<uchar>(i, j);
			output.at<uchar>(i, j) = lut[input_value];
		}
	}
}


int main()
{
	// 读取灰度图
	cv::Mat img = cv::imread("img/img1.png", cv::IMREAD_GRAYSCALE);

	if (img.empty())
	{
		std::cerr << "Error: Image not found" << '\n';
		return 1;
	}

	auto output = cv::Mat();
	try
	{
		invert_grayscale(img, output);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}

	try
	{
		log_transform(img, output);
	}
	catch (const std::exception& e)
	{
		std::cerr << "Error: " << e.what() << '\n';
		return 1;
	}

	float c = 0.8f;
	for (int i = 0; i < 5; i++)
	{
		c += 0.1f;
		float gamma = 1.8f;
		for (int j = 0; j < 5; j++)
		{
			gamma += 0.1f;
			try
			{
				power_transform_with_save(img, output, c, gamma);
			}
			catch (const std::exception& e)
			{
				std::cerr << "Error: " << e.what() << '\n';
				return 1;
			}
		}
	}


	constexpr int LOOPTIME = 1000;

	// 用于存储输出
	cv::Mat output_basic, output_optimized;

	// 统计预热消耗时间
	auto preheat_start = std::chrono::high_resolution_clock::now();
	power_transform(img, output_basic, 1.0f, 2.0f);
	power_transform_better(img, output_optimized, 1.0f, 2.0f);
	auto preheat_end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> preheat_ms = preheat_end - preheat_start;

	double total_time_basic = 0.0;
	double total_time_better = 0.0;

	for (int i = 0; i < LOOPTIME; ++i)
	{
		float c = 1.0f;
		float gamma = 2.0f;

		// 基础版本测试
		{
			cv::Mat temp;
			auto start = std::chrono::high_resolution_clock::now();
			power_transform(img, temp, c, gamma);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> ms = end - start;
			total_time_basic += ms.count();
		}

		// 优化版本测试
		{
			cv::Mat temp;
			auto start = std::chrono::high_resolution_clock::now();
			power_transform_better(img, temp, c, gamma);
			auto end = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> ms = end - start;
			total_time_better += ms.count();
		}
	}

	std::print("预热消耗时间: {:.3f} ms\n", preheat_ms.count());
	std::print("power_transform 平均耗时: {:.3f} ms\n", total_time_basic / LOOPTIME);
	std::print("power_transform_better 平均耗时: {:.3f} ms\n", total_time_better / LOOPTIME);
}
