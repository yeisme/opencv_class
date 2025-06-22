add_executable(class7_learn_main
    main.cpp
    utils.cpp
)

if(MSVC)
    target_compile_options(class7_learn_main PRIVATE /openmp /arch:AVX512)
endif()

# 链接共享配置到两个可执行文件
target_include_directories(class7_learn_main PRIVATE ${OpenCV_INCLUDES_DIRS})
target_link_libraries(class7_learn_main PRIVATE ${OpenCV_LIBS})
target_include_directories(class7_learn_main PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_custom_target(class7_copy_img_files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/img
    ${CMAKE_CURRENT_BINARY_DIR}/img
    COMMENT "正在复制图像文件到构建目录..."
)

add_dependencies(class7_learn_main class7_copy_img_files)
