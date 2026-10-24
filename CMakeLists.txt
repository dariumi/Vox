cmake_minimum_required(VERSION 3.18)
project(vox VERSION 0.19.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(VOX_NATIVE "Оптимизировать под CPU этой машины (-march=native)" OFF)
option(VOX_PYTHON "Собрать Python-модуль (нужен nanobind)" OFF)
option(VOX_TESTS "Собрать тесты" ON)
option(VOX_VULKAN "GPU-бэкенд на Vulkan (нужны заголовки Vulkan; шейдеры уже скомпилированы)" ON)

if(VOX_NATIVE AND NOT MSVC)
  add_compile_options(-march=native)
endif()
if(NOT MSVC)
  add_compile_options(-Wall -Wextra -Wno-psabi)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-psabi)
  endif()
endif()

# Ядро: всё, кроме обёрток. Статическая библиотека, из неё собираются
# libvox (C ABI), CLI, тесты и Python-модуль.
add_library(vox_core STATIC
  src/formats.cpp
  src/voice.cpp src/profile.cpp src/markup.cpp
  src/tokenizer.cpp
  src/kernels.cpp src/resources.cpp src/tune.cpp src/family.cpp src/f5_ops.cpp src/f5.cpp
  src/qwen3.cpp src/qwen3_voice.cpp
  src/backend_qwen3.cpp
  src/backend_f5.cpp src/backend_kokoro.cpp src/kokoro.cpp src/nn.cpp src/cosyvoice.cpp src/cosyvoice_front.cpp src/backend_cosyvoice.cpp src/hf_tokenizer.cpp src/chatterbox.cpp src/backend_chatterbox.cpp src/kokoro_g2p.cpp src/kokoro_g2p_en.cpp
  src/backend_null.cpp
  src/engine.cpp)
if(VOX_VULKAN)
  find_package(Vulkan)
endif()
if(VOX_VULKAN AND Vulkan_FOUND)
  target_sources(vox_core PRIVATE src/gpu_vulkan.cpp src/gpu_codec.cpp src/gpu_f5.cpp)
  set_source_files_properties(src/gpu_vulkan.cpp src/gpu_codec.cpp src/gpu_f5.cpp PROPERTIES COMPILE_OPTIONS "-Wno-missing-field-initializers")
  target_link_libraries(vox_core PUBLIC Vulkan::Vulkan)
  message(STATUS "vox: Vulkan backend enabled")
else()
  target_sources(vox_core PRIVATE src/gpu_stub.cpp)
  message(STATUS "vox: Vulkan backend disabled")
endif()
find_package(Threads REQUIRED)
target_link_libraries(vox_core PUBLIC Threads::Threads)
target_include_directories(vox_core PUBLIC include)
target_compile_definitions(vox_core PUBLIC VOX_BUILD VOX_VERSION_STRING="${PROJECT_VERSION}")

# Публичная разделяемая библиотека с C ABI.
add_library(vox SHARED src/abi_export.cpp)
target_link_libraries(vox PRIVATE vox_core)
target_include_directories(vox PUBLIC include)
set_target_properties(vox PROPERTIES OUTPUT_NAME vox VERSION ${PROJECT_VERSION} SOVERSION 0)
if(NOT WIN32 AND NOT APPLE)
  target_link_options(vox PRIVATE -Wl,--whole-archive $<TARGET_FILE:vox_core> -Wl,--no-whole-archive)
elseif(APPLE)
  target_link_options(vox PRIVATE -Wl,-force_load,$<TARGET_FILE:vox_core>)
elseif(MSVC)
  target_link_options(vox PRIVATE /WHOLEARCHIVE:vox_core)
endif()

add_executable(vox_cli cli/main.cpp)
target_link_libraries(vox_cli PRIVATE vox_core)
set_target_properties(vox_cli PROPERTIES OUTPUT_NAME vox)

if(VOX_TESTS)
  enable_testing()
  add_executable(vox_tests tests/test_core.cpp)
  target_link_libraries(vox_tests PRIVATE vox_core)
  add_test(NAME core COMMAND vox_tests WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

if(VOX_PYTHON)
  find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
  execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_vox NB_STATIC python/bindings.cpp)
  target_link_libraries(_vox PRIVATE vox_core)
  install(TARGETS _vox LIBRARY DESTINATION vox)
endif()
