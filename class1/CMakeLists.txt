add_executable(class1_learn_test "test.cpp")
add_executable(class1_learn_main "main.cpp")

# 链接共享配置到两个可执行文件
target_link_libraries(class1_learn_test PRIVATE shared_config)
target_link_libraries(class1_learn_main PRIVATE shared_config)

# 创建自定义目标用于复制图像文件
add_custom_target(copy_img_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/img
    ${CMAKE_CURRENT_BINARY_DIR}/img
    COMMENT "正在复制图像文件到构建目录..."
)

# 添加依赖关系，确保在构建可执行文件之前复制图像
add_dependencies(class1_learn_main copy_img_files)
add_dependencies(class1_learn_test copy_img_files)
