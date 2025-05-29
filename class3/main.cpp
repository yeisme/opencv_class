#include <opencv2/opencv.hpp>
#include <print>
#include <vector>
#include <chrono> // Required for timing
#include <omp.h> // Added for OpenMP

namespace ye_std
{
	using namespace cv;

	/**
	 * @brief
	 * 平均滤波通过计算邻域像素的算术平均值来替换中心像素,
	 * 核函数为均匀权重：每个位置权重 = 1/(核宽度×核高度)
	 * 
	 * @param src 
	 * @param dst 
	 * @param ksize 
	 * @param anchor 
	 * @param borderType 
	 */
	void blur(InputArray src, OutputArray dst, Size ksize, Point anchor = Point(-1, -1),
	          int borderType = BORDER_DEFAULT)
	{
		Mat input = src.getMat();

		dst.create(input.size(), input.type());
		Mat output = dst.getMat();

		// 验证核大小
		if (ksize.width <= 0 || ksize.height <= 0 ||
			ksize.width % 2 == 0 || ksize.height % 2 == 0)
		{
			throw std::invalid_argument("Kernel size must be positive and odd");
		}

		// 设置锚点（默认为核中心）
		if (anchor.x == -1 || anchor.y == -1)
		{
			anchor.x = ksize.width / 2;
			anchor.y = ksize.height / 2;
		}

		// 计算边界填充
		int top = anchor.y;
		int bottom = ksize.height - anchor.y - 1;
		int left = anchor.x;
		int right = ksize.width - anchor.x - 1;

		// 边界填充
		// 使用 copyMakeBorder 进行边界填充
		// 支持多种填充方式：常数、反射、复制等
		Mat padded;
		copyMakeBorder(input, padded, top, bottom, left, right, borderType);


		const float kernel_size_inv = 1.0f / (ksize.width * ksize.height);
		const int vectorSize = 64; // AVX512 可以处理 64 个 uint8


#pragma omp parallel for schedule(dynamic) if(input.rows * input.cols > 10000)
		for (int row = 0; row < input.rows; row++)
		{
			if (input.channels() == 1)
			{
				int col = 0;
				const int step = 16; // 每次处理16个像素

				// AVX512 向量化处理
				for (; col <= input.cols - step; col += step)
				{
					__m512i sum = _mm512_setzero_si512();

					// 遍历滤波核
					for (int kr = 0; kr < ksize.height; kr++)
					{
						const uchar* row_ptr = padded.ptr<uchar>(row + kr);
						for (int kc = 0; kc < ksize.width; kc++)
						{
							// 加载16个像素并转换为32位整数
							__m128i pixels8 = _mm_loadu_si128((__m128i*)(row_ptr + col + kc));
							__m512i pixels32 = _mm512_cvtepu8_epi32(pixels8);
							sum = _mm512_add_epi32(sum, pixels32);
						}
					}

					// 转换为浮点数，计算平均值，转换回整数
					__m512 sum_f = _mm512_cvtepi32_ps(sum);
					__m512 avg = _mm512_mul_ps(sum_f, _mm512_set1_ps(kernel_size_inv));
					__m512i result32 = _mm512_cvtps_epi32(avg);

					// 饱和转换为uint8并存储
					__m128i result8 = _mm512_cvtusepi32_epi8(result32);
					_mm_storeu_si128((__m128i*)(output.ptr<uchar>(row) + col), result8);
				}

				// 处理剩余像素
				for (; col < input.cols; col++)
				{
					float sum = 0;
					for (int kr = 0; kr < ksize.height; kr++)
					{
						for (int kc = 0; kc < ksize.width; kc++)
						{
							sum += padded.at<uchar>(row + kr, col + kc);
						}
					}
					output.at<uchar>(row, col) = saturate_cast<uchar>(sum * kernel_size_inv);
				}
			}
		}
	}

	/**
	 * @brief 
	 * @param src 
	 * @param dst 
	 * @param ksize 
	 */
	void median_blur(InputArray src, OutputArray dst, int ksize)
	{
		Mat input = src.getMat();
		dst.create(input.size(), input.type());
		Mat output = dst.getMat();

		// 验证核大小
		if (ksize <= 0 || ksize % 2 == 0)
		{
			throw std::invalid_argument("Kernel size must be positive and odd");
		}

		// 计算边界填充
		int pad = ksize / 2;

		// 边界填充
		Mat padded;
		copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_DEFAULT);

#pragma omp parallel for schedule(dynamic) if(input.rows * input.cols > 10000)
		for (int row = 0; row < input.rows; row++)
		{
			for (int col = 0; col < input.cols; col++)
			{
				if (input.channels() == 1)
				{
					std::vector<uchar> neighborhood;
					for (int kr = 0; kr < ksize; kr++)
					{
						for (int kc = 0; kc < ksize; kc++)
						{
							neighborhood.push_back(padded.at<uchar>(row + kr, col + kc));
						}
					}
					std::sort(neighborhood.begin(), neighborhood.end());
					output.at<uchar>(row, col) = neighborhood[neighborhood.size() / 2];
				}
				else if (input.channels() == 3)
				{
					throw std::invalid_argument("Unimplemented for 3-channel images\n");
				}
			}
		}
	}

	void gaussian_blur(InputArray src, OutputArray dst, Size ksize,
	                   double sigmaX, double sigmaY = 0,
	                   int borderType = BORDER_DEFAULT,
	                   AlgorithmHint hint = cv::ALGO_HINT_DEFAULT)
	{
		Mat input = src.getMat();
		dst.create(input.size(), input.type());
		Mat output = dst.getMat();

		// 验证核大小
		if (ksize.width <= 0 || ksize.height <= 0 ||
			ksize.width % 2 == 0 || ksize.height % 2 == 0)
		{
			throw std::invalid_argument("Kernel size must be positive and odd");
		}

		if (sigmaX <= 0)
		{
			sigmaX = 0.3 * ((ksize.width - 1) * 0.5 - 1) + 0.8;
		}
		if (sigmaY <= 0)
		{
			sigmaY = 0.3 * ((ksize.height - 1) * 0.5 - 1) + 0.8;
		}

		// 创建高斯核
		Mat kernelX = getGaussianKernel(ksize.width, sigmaX, CV_64F);
		Mat kernelY = getGaussianKernel(ksize.height, sigmaY, CV_64F);
		Mat kernel = kernelX * kernelY.t(); // 2D kernel

		// 计算边界填充
		int pad_w = ksize.width / 2;
		int pad_h = ksize.height / 2;

		Mat padded;
		copyMakeBorder(input, padded, pad_h, pad_h, pad_w, pad_w, borderType);

#pragma omp parallel for schedule(dynamic) if(input.rows * input.cols > 10000)
		for (int row = 0; row < input.rows; row++)
		{
			for (int col = 0; col < input.cols; col++)
			{
				if (input.channels() == 1)
				{
					double sum = 0.0;
					for (int kr = 0; kr < ksize.height; kr++)
					{
						for (int kc = 0; kc < ksize.width; kc++)
						{
							sum += padded.at<uchar>(row + kr, col + kc) * kernel.at<double>(kr, kc);
						}
					}
					output.at<uchar>(row, col) = saturate_cast<uchar>(sum);
				}
				else if (input.channels() == 3)
				{
					throw std::invalid_argument("Unimplemented for 3-channel images\n");
				}
			}
		}
	}
}


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
	CV_Assert(!src.empty() && src.channels() == 1);
	cv::Mat noise = cv::Mat(src.size(), CV_8SC1);
	// 生成符合高斯分布的随机噪声
	cv::randn(noise, mean, stddev);

	src.copyTo(dst);
	// 添加噪声
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			// 将噪声加到原始像素值上
			int value = static_cast<int>(dst.at<uchar>(i, j)) + noise.at<char>(i, j);
			dst.at<uchar>(i, j) = static_cast<uchar>(std::clamp(value, 0, 255));
		}
	}
}


int main()
{
	cv::Mat img = cv::imread("img/img1.png", cv::IMREAD_GRAYSCALE);

	CV_Assert(!img.empty());

	cv::imshow("原始灰度图", img);

	// 高斯噪声
	cv::Mat noisy_img1, noisy_img2, noisy_img3;
	add_gaussian_noise(img, noisy_img1, 0, 10);
	add_gaussian_noise(img, noisy_img2, 0, 25); // Using noisy_img2 for tests
	add_gaussian_noise(img, noisy_img3, 0, 50);

	cv::imwrite("img/noisy_light.png", noisy_img1);
	cv::imwrite("img/noisy_medium.png", noisy_img2);
	cv::imwrite("img/noisy_heavy.png", noisy_img3);
	std::print("已添加高斯噪声并保存结果\n");

	// 降噪测试
	try
	{
		const int num_iterations = 1000; // Number of iterations for averaging time
		const int k_val_3 = 3;
		const int k_val_5 = 5;
		cv::Size kernel_size_k3(k_val_3, k_val_3);
		cv::Size kernel_size_k5(k_val_5, k_val_5);
		double sigma = 1.0;

		cv::Mat ye_blur_out_k3, cv_blur_out_k3;
		cv::Mat ye_blur_out_k5, cv_blur_out_k5;
		cv::Mat ye_median_out_k3, cv_median_out_k3;
		cv::Mat ye_median_out_k5, cv_median_out_k5;
		cv::Mat ye_gaussian_out_k3, cv_gaussian_out_k3;
		cv::Mat ye_gaussian_out_k5, cv_gaussian_out_k5;

		std::chrono::microseconds total_duration_ye(0);
		std::chrono::microseconds total_duration_cv(0);
		auto start_time = std::chrono::high_resolution_clock::now();
		auto end_time = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

		std::print("Running tests, timing for k=5 with {} iterations...\n", num_iterations);

		// --- ye_std::blur vs cv::blur ---
		std::print("--- Testing blur ---\n");
		// k=3
		ye_std::blur(noisy_img2, ye_blur_out_k3, kernel_size_k3);
		cv::imwrite("img/ye_blur_out_k3.png", ye_blur_out_k3);
		cv::blur(noisy_img2, cv_blur_out_k3, kernel_size_k3);
		cv::imwrite("img/cv_blur_out_k3.png", cv_blur_out_k3);
		// k=5 (with timing)
		total_duration_ye = std::chrono::microseconds(0);
		for (int i = 0; i < num_iterations; ++i)
		{
			start_time = std::chrono::high_resolution_clock::now();
			ye_std::blur(noisy_img2, ye_blur_out_k5, kernel_size_k5);
			end_time = std::chrono::high_resolution_clock::now();
			total_duration_ye += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		}
		duration = total_duration_ye / num_iterations;
		std::print("Average ye_std::blur time (k=5): {} us\n", duration.count());
		cv::imwrite("img/ye_blur_out_k5.png", ye_blur_out_k5);

		total_duration_cv = std::chrono::microseconds(0);
		for (int i = 0; i < num_iterations; ++i)
		{
			start_time = std::chrono::high_resolution_clock::now();
			cv::blur(noisy_img2, cv_blur_out_k5, kernel_size_k5);
			end_time = std::chrono::high_resolution_clock::now();
			total_duration_cv += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		}
		duration = total_duration_cv / num_iterations;
		std::print("Average cv::blur time (k=5): {} us\n", duration.count());
		cv::imwrite("img/cv_blur_out_k5.png", cv_blur_out_k5);

		// --- ye_std::median_blur vs cv::medianBlur ---
		std::print("--- Testing median_blur ---\n");
		// k=3
		ye_std::median_blur(noisy_img2, ye_median_out_k3, k_val_3);
		cv::imwrite("img/ye_median_blur_out_k3.png", ye_median_out_k3);
		cv::medianBlur(noisy_img2, cv_median_out_k3, k_val_3);
		cv::imwrite("img/cv_median_blur_out_k3.png", cv_median_out_k3);
		// k=5 (with timing)
		total_duration_ye = std::chrono::microseconds(0);
		for (int i = 0; i < num_iterations; ++i)
		{
			start_time = std::chrono::high_resolution_clock::now();
			ye_std::median_blur(noisy_img2, ye_median_out_k5, k_val_5);
			end_time = std::chrono::high_resolution_clock::now();
			total_duration_ye += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		}
		duration = total_duration_ye / num_iterations;
		std::print("Average ye_std::median_blur time (k=5): {} us\n", duration.count());
		cv::imwrite("img/ye_median_blur_out_k5.png", ye_median_out_k5);

		total_duration_cv = std::chrono::microseconds(0);
		for (int i = 0; i < num_iterations; ++i)
		{
			start_time = std::chrono::high_resolution_clock::now();
			cv::medianBlur(noisy_img2, cv_median_out_k5, k_val_5);
			end_time = std::chrono::high_resolution_clock::now();
			total_duration_cv += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		}
		duration = total_duration_cv / num_iterations;
		std::print("Average cv::medianBlur time (k=5): {} us\n", duration.count());
		cv::imwrite("img/cv_median_blur_out_k5.png", cv_median_out_k5);

		// --- ye_std::gaussian_blur vs cv::GaussianBlur ---
		std::print("--- Testing gaussian_blur ---\n");
		// k=3
		ye_std::gaussian_blur(noisy_img2, ye_gaussian_out_k3, kernel_size_k3, sigma);
		cv::imwrite("img/ye_gaussian_blur_out_k3.png", ye_gaussian_out_k3);
		cv::GaussianBlur(noisy_img2, cv_gaussian_out_k3, kernel_size_k3, sigma);
		cv::imwrite("img/cv_gaussian_blur_out_k3.png", cv_gaussian_out_k3);
		// k=5 (with timing)
		total_duration_ye = std::chrono::microseconds(0);
		for (int i = 0; i < num_iterations; ++i)
		{
			start_time = std::chrono::high_resolution_clock::now();
			ye_std::gaussian_blur(noisy_img2, ye_gaussian_out_k5, kernel_size_k5, sigma);
			end_time = std::chrono::high_resolution_clock::now();
			total_duration_ye += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		}
		duration = total_duration_ye / num_iterations;
		std::print("Average ye_std::gaussian_blur time (k=5): {} us\n", duration.count());
		cv::imwrite("img/ye_gaussian_blur_out_k5.png", ye_gaussian_out_k5);

		total_duration_cv = std::chrono::microseconds(0);
		for (int i = 0; i < num_iterations; ++i)
		{
			start_time = std::chrono::high_resolution_clock::now();
			cv::GaussianBlur(noisy_img2, cv_gaussian_out_k5, kernel_size_k5, sigma);
			end_time = std::chrono::high_resolution_clock::now();
			total_duration_cv += std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
		}
		duration = total_duration_cv / num_iterations;
		std::print("Average cv::GaussianBlur time (k=5): {} us\n", duration.count());
		cv::imwrite("img/cv_gaussian_blur_out_k5.png", cv_gaussian_out_k5);

		// Display results (optional)
		// cv::imshow("ye_std::blur_k3", ye_blur_out_k3);
		// cv::imshow("cv::blur_k3", cv_blur_out_k3);
		// cv::imshow("ye_std::blur_k5", ye_blur_out_k5);
		// cv::imshow("cv::blur_k5", cv_blur_out_k5);
	}
	catch (const std::exception& e)
	{
		std::print("Exception: {}\n", e.what());
	}

	cv::waitKey(0);
	return 0;
}
