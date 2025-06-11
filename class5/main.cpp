#include <filesystem>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <print>
#include <string>

// 辅助函数，用于显示图像并保存
void display_and_save(const cv::Mat &image,
                      const std::string &window_name_chinese,
                      const std::string &save_file_suffix_chinese,
                      const std::string &save_path_base) {
  if (image.empty()) {
    std::print("警告: 图像 '{}' 为空，无法显示或保存。\n", window_name_chinese);
    return;
  }
  cv::imshow(window_name_chinese, image);  // 使用中文窗口名
  std::string full_save_path =
      "out/" + save_path_base + "_" + save_file_suffix_chinese + ".png";
  if (!cv::imwrite(full_save_path, image)) {
    std::print("警告: 无法保存图像到 '{}'\n", full_save_path);
  } else {
    std::print("图像已保存到: {}\n", full_save_path);
  }
}

int main() {
  std::string image_path = "img/image.png";  // 请确保此路径下有图像文件
  std::string output_base_name =
      "形态学操作结果";  // 用于保存文件的前缀（中文）

  std::filesystem::create_directories("out");  // 确保输出目录存在

  // 1. 调入图像
  cv::Mat original_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

  if (original_image.empty()) {
    std::print("错误: 无法加载图像 '{}'. 请检查路径是否正确。\n", image_path);
    return -1;
  }
  display_and_save(original_image, "原始灰度图", "原始灰度图",
                   output_base_name);

  // 2. 采用OTSU方法，对图像进行二值化处理
  cv::Mat otsu_image;
  cv::threshold(original_image, otsu_image, 0, 255,
                cv::THRESH_BINARY + cv::THRESH_OTSU);
  display_and_save(otsu_image, "OTSU二值化图", "OTSU二值化", output_base_name);

  // 3. 设置结构元素
  // 您可以尝试不同的形状 (cv::MORPH_RECT, cv::MORPH_CROSS, cv::MORPH_ELLIPSE)
  // 和大小 (例如 cv::Size(3,3), cv::Size(5,5))
  int morph_size = 2;
  cv::Mat structuring_element = cv::getStructuringElement(
      cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1),
      cv::Point(morph_size, morph_size));
  std::print("使用的结构元素类型: MORPH_RECT, 大小: {}x{}\n",
             2 * morph_size + 1, 2 * morph_size + 1);

  // 4. 对得到的二值图像进行腐蚀运算
  cv::Mat eroded_image;
  cv::erode(otsu_image, eroded_image, structuring_element);
  display_and_save(eroded_image, "腐蚀运算后", "腐蚀运算", output_base_name);

  // 5. 对得到的二值图像进行膨胀运算
  cv::Mat dilated_image;
  cv::dilate(otsu_image, dilated_image, structuring_element);
  display_and_save(dilated_image, "膨胀运算后", "膨胀运算", output_base_name);

  // 6. 对得到的二值图像进行开运算 (先腐蚀后膨胀)
  cv::Mat opened_image;
  cv::morphologyEx(otsu_image, opened_image, cv::MORPH_OPEN,
                   structuring_element);
  display_and_save(opened_image, "开运算后", "开运算", output_base_name);

  // 7. 对得到的二值图像进行闭运算 (先膨胀后腐蚀)
  cv::Mat closed_image;
  cv::morphologyEx(otsu_image, closed_image, cv::MORPH_CLOSE,
                   structuring_element);
  display_and_save(closed_image, "闭运算后", "闭运算", output_base_name);

  // 8. 将两种处理方法的结果作比较
  // "两种处理方法"的比较可以有多种理解。
  // 这里我们将OTSU二值图与开运算和闭运算的结果并列显示，以便比较它们对噪声和孔洞的处理。
  // 腐蚀和膨胀是基础操作，开闭运算是它们的组合，通常用于更具体的目的（如去噪、连接断裂）。

  // 比较1: OTSU vs 开运算 (开运算通常用于去除小的噪声点)
  // 比较2: OTSU vs 闭运算 (闭运算通常用于填充小的孔洞)
  // 比较3: 腐蚀 vs 膨胀 (基础形态学操作的效果)

  std::print("\n处理完成。所有图像已显示并尝试保存到 'out/' 目录中。\n");
  std::print(
      "您可以比较 'OTSU二值化图' 与 '开运算后' (去除小对象/噪声) 以及 "
      "'闭运算后' (填充小孔洞) 的效果。\n");
  std::print(
      "同时也可以观察 '腐蚀运算后' (前景缩小) 和 '膨胀运算后' (前景扩大) "
      "的效果。\n");

  cv::waitKey(0);           // 等待按键后关闭所有窗口
  cv::destroyAllWindows();  // 确保所有窗口都关闭
  return 0;
}