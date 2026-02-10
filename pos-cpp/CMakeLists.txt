# POS Core - 分层记忆系统 v2.0 构建配置
cmake_minimum_required(VERSION 3.16)
project(POSCore VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 基本依赖
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

# 查找依赖
find_package(yaml-cpp QUIET)
find_package(RocksDB QUIET)

# 检查Boost.Geometry
find_package(Boost QUIET COMPONENTS system)

# FetchContent
include(FetchContent)

# JSON库
FetchContent_Declare(json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2
    GIT_SHALLOW TRUE
)

# Crow
FetchContent_Declare(crow
    GIT_REPOSITORY https://github.com/CrowCpp/Crow.git
    GIT_TAG v1.0+5
    GIT_SHALLOW TRUE
)

# HNSW - 向量索引
FetchContent_Declare(hnsw
    GIT_REPOSITORY https://github.com/nmslib/hnswlib.git
    GIT_TAG v0.7.0
    GIT_SHALLOW TRUE
)

# Folly - 用于序列化等
FetchContent_Declare(folly
    GIT_REPOSITORY https://github.com/facebook/folly.git
    GIT_TAG v2024.01.01.00
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(json crow hnsw)

# 简化版本源文件
set(CORE_SOURCES_SIMPLE
    src/core/common/types.cpp
    src/ml/ml_service_client.cpp
)

# 完整版本源文件 - 包含四层记忆系统
set(CORE_SOURCES_FULL
    src/core/common/types.cpp
    src/core/ontology/ontology_graph.cpp
    src/core/ontology/ontology_graph_part2.cpp
    src/core/ontology/ontology_graph_part3.cpp
    src/core/memory/hierarchical_memory_part1.cpp
    src/core/memory/hierarchical_memory_part2.cpp
    src/core/memory/hierarchical_memory_part3.cpp
    src/core/memory/hierarchical_memory_part4.cpp
    src/core/temporal/spatiotemporal_engine.cpp
    src/core/reasoning/reasoning_engine.cpp
    src/core/pos_core_engine.cpp
    src/ml/ml_service_client.cpp
)

# 创建简化版可执行文件
add_executable(pos_server_simple
    ${CORE_SOURCES_SIMPLE}
    src/api/http_server_simple.cpp
)

target_include_directories(pos_server_simple PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(pos_server_simple PRIVATE
    Threads::Threads
    CURL::libcurl
    nlohmann_json::nlohmann_json
    Crow::Crow
)

# 如果有完整依赖，构建完整版本
if(RocksDB_FOUND AND yaml-cpp_FOUND AND Boost_FOUND)
    message(STATUS "Building full POS Core v2.0 with Quad-Layer Memory System")
    
    add_executable(pos_server
        ${CORE_SOURCES_FULL}
        src/api/http_server.cpp
    )
    
    target_include_directories(pos_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${hnsw_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
    )
    
    target_link_libraries(pos_server PRIVATE
        Threads::Threads
        CURL::libcurl
        yaml-cpp
        RocksDB::rocksdb
        nlohmann_json::nlohmann_json
        Crow::Crow
        ${Boost_LIBRARIES}
    )
    
    target_compile_definitions(pos_server PRIVATE
        ENABLE_HIERARCHICAL_MEMORY=1
        ENABLE_HNSW_INDEX=1
        ENABLE_SPATIAL_INDEX=1
    )
    
    # 四层记忆系统测试
    add_executable(test_quad_layer_memory
        src/core/memory/hierarchical_memory_part1.cpp
        src/core/memory/hierarchical_memory_part2.cpp
        src/core/memory/hierarchical_memory_part3.cpp
        src/core/memory/hierarchical_memory_part4.cpp
        tests/test_quad_layer_memory.cpp
    )
    
    target_include_directories(test_quad_layer_memory PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${hnsw_SOURCE_DIR}
        ${Boost_INCLUDE_DIRS}
    )
    
    target_link_libraries(test_quad_layer_memory PRIVATE
        Threads::Threads
        RocksDB::rocksdb
        nlohmann_json::nlohmann_json
        ${Boost_LIBRARIES}
    )
    
else()
    message(WARNING "Missing dependencies for full build:")
    if(NOT RocksDB_FOUND)
        message(WARNING "  - RocksDB not found")
    endif()
    if(NOT yaml-cpp_FOUND)
        message(WARNING "  - yaml-cpp not found")
    endif()
    if(NOT Boost_FOUND)
        message(WARNING "  - Boost not found")
    endif()
    message(WARNING "Only building simple version. Install dependencies for full build.")
    message(WARNING "  macOS: brew install rocksdb yaml-cpp boost")
    message(WARNING "  Ubuntu: sudo apt-get install librocksdb-dev libyaml-cpp-dev libboost-all-dev")
endif()

message(STATUS "POS Core v${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
