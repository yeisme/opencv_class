add_executable(class4_learn_main "main.cpp")

if (MSVC) 
    target_compile_options(class4_learn_main PRIVATE /openmp /arch:AVX512 )
endif()

# 链接共享配置到两个可执行文件
target_include_directories(class4_learn_main PRIVATE ${OpenCV_INCLUDES_DIRS})
target_link_libraries(class4_learn_main PRIVATE ${OpenCV_LIBS})

add_custom_target(class4_copy_img_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/img
    ${CMAKE_CURRENT_BINARY_DIR}/img
    COMMENT "正在复制图像文件到构建目录..."
)

add_dependencies(class4_learn_main class4_copy_img_files)
