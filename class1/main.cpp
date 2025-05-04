#include <filesystem>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <print>

int main()
{
	// 1．use imread() read a image flower.tif
	auto img = cv::imread("img/flower.tif");
	if (img.empty())
	{
		std::cerr << "Could not read the image\n";
		return -1;
	}

	// 2．use imwrite() to save the original image
	cv::imshow("原始图像", img);
	cv::waitKey(0);

	// 3．利用Mat 类的ptr, at等操作，对图像像素值进行读取和修改
	// 创建一个副本用于修改
	cv::Mat modified_img = img.clone();

	// 方法一：使用ptr方法修改像素值
	for (int y = 0; y < modified_img.rows; y++)
	{
		// 获取当前行的指针
		uchar* row_ptr = modified_img.ptr<uchar>(y);

		// 根据行号判断是偶数行还是奇数行
		uchar value = (y & 1) ? 0 : 255;

		// 修改整行的像素值
		for (int x = 0; x < modified_img.cols * modified_img.channels(); x++)
		{
			row_ptr[x] = value;
		}
	}

	// 显示修改后的图像（使用ptr方法）
	cv::imshow("使用ptr方法修改后的图像", modified_img);

	// 方法二：使用at方法修改像素值
	cv::Mat modified_img2 = img.clone();

	for (int y = 0; y < modified_img2.rows; y++)
	{
		// 根据行号判断是偶数行还是奇数行
		uchar value = (y & 1) ? 0 : 255;

		for (int x = 0; x < modified_img2.cols; x++)
		{
			// 对于彩色图像(3通道)，需要修改每个通道
			if (modified_img2.channels() == 3)
			{
				modified_img2.at<cv::Vec3b>(y, x) = cv::Vec3b(value, value, value);
			}
			// 对于灰度图像(单通道)
			else if (modified_img2.channels() == 1)
			{
				modified_img2.at<uchar>(y, x) = value;
			}
		}
	}

	// 显示修改后的图像（使用at方法）
	cv::imshow("使用at方法修改后的图像", modified_img2);
	cv::waitKey(0);

	// 保存修改后的图像
	cv::imwrite("img/modified_flower.tif", modified_img2);

	// 4. 利用imwrite()函数来压缩这幅图象，将其保存为一幅压缩了像素的jpg文件,设为flower.jpg。
	bool compression_success = cv::imwrite("img/flower.jpg", img, {cv::IMWRITE_JPEG_QUALITY, 50});
	if (compression_success)
	{
		std::print("压缩图片成功，设置了50%的JPEG质量\n");

		// 获取原图和压缩图的文件大小进行比较
		auto original_size = std::filesystem::file_size("img/flower.tif");
		auto compressed_size = std::filesystem::file_size("img/flower.jpg");
		std::print("原图大小: {} 字节\n", original_size);
		std::print("压缩后大小: {} 字节\n", compressed_size);
		std::print("压缩率: {:.2f}%\n", (1.0 - static_cast<double>(compressed_size) / original_size) * 100);
	}
	else
	{
		std::print("压缩图片失败\n");
	}


	return 0;
}
