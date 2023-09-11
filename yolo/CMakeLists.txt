

set (EXTRA_INCLUDE "")
set (EXTRA_LIBS "")
set (EXTRA_OPTIONS "")

if(MNN_CUDA_PROFILE)
    set(CUDA_MIN_VERSION "9.0")
    find_package(CUDA ${CUDA_MIN_VERSION})
    MESSAGE("yolo message:CUDA_INCLUDE_DIRS:${CUDA_INCLUDE_DIRS}")
    link_directories(${CUDA_INCLUDE_DIRS}/../lib/
        ${CUDA_INCLUDE_DIRS}/../lib64/)
    set(EXTRA_INCLUDE ${CUDA_INCLUDE_DIRS})
    set(EXTRA_LIBS  -lnvToolsExt)
    set(EXTRA_OPTIONS -DMNN_CUDA_PROFILE)
endif()



add_executable(yolov5.out ${CMAKE_CURRENT_LIST_DIR}/yolov5.cpp ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/revertMNNModel.cpp ${CMAKE_CURRENT_SOURCE_DIR}/yolo/ocv)
target_include_directories(yolov5.out PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/cpp/ ${CMAKE_CURRENT_SOURCE_DIR}/tools/ ${CMAKE_CURRENT_SOURCE_DIR}/yolo/ocv ${EXTRA_INCLUDE})
target_compile_options(yolov5.out PRIVATE ${EXTRA_OPTIONS})
target_link_libraries(yolov5.out ${MNN_DEPS}  ${EXTRA_LIBS})



if (MSVC AND NOT MNN_BUILD_SHARED_LIBS)
  foreach (DEPEND ${MNN_DEPS})
    target_link_options(yolov5.out PRIVATE /WHOLEARCHIVE:$<TARGET_FILE:${DEPEND}>)
  endforeach ()
endif()