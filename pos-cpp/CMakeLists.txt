# POS Core - 简化构建配置
cmake_minimum_required(VERSION 3.16)
project(POSCore VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 基本依赖
find_package(Threads REQUIRED)
find_package(CURL REQUIRED)

# 查找yaml-cpp和RocksDB
find_package(yaml-cpp QUIET)
find_package(RocksDB QUIET)

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

FetchContent_MakeAvailable(json crow)

# 源文件 - 核心
set(CORE_SOURCES
    src/core/common/types.cpp
    # 简化版本 - 暂时不编译需要RocksDB的代码
    # src/core/ontology/ontology_graph.cpp
    # src/core/memory/memory_store.cpp
    # src/core/reasoning/reasoning_engine.cpp
    # src/core/pos_core_engine.cpp
    src/ml/ml_service_client.cpp
)

# 创建简化版可执行文件
add_executable(pos_server_simple
    ${CORE_SOURCES}
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

# 如果有RocksDB，构建完整版本
if(RocksDB_FOUND AND yaml-cpp_FOUND)
    message(STATUS "Building full POS Core with RocksDB")
    
    add_executable(pos_server
        src/core/common/types.cpp
        src/core/ontology/ontology_graph.cpp
        src/core/memory/memory_store.cpp
        src/core/reasoning/reasoning_engine.cpp
        src/core/pos_core_engine.cpp
        src/ml/ml_service_client.cpp
        src/api/http_server.cpp
    )
    
    target_include_directories(pos_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    
    target_link_libraries(pos_server PRIVATE
        Threads::Threads
        CURL::libcurl
        yaml-cpp
        RocksDB::rocksdb
        nlohmann_json::nlohmann_json
        Crow::Crow
    )
else()
    message(WARNING "RocksDB or yaml-cpp not found, only building simple version")
endif()

message(STATUS "POS Core v${PROJECT_VERSION}")
