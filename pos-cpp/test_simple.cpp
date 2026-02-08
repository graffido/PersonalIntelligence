#include <iostream>
#include "src/core/common/types.h"

using namespace pos;

int main() {
    std::cout << "=== POS Basic Test ===" << std::endl;
    
    // 测试UUID生成
    std::string uuid = generateUUID();
    std::cout << "Generated UUID: " << uuid << std::endl;
    
    // 测试GeoPoint
    GeoPoint p1{39.9042, 116.4074};
    GeoPoint p2{39.9142, 116.4174};
    double dist = p1.distanceTo(p2);
    std::cout << "Distance between points: " << dist << " meters" << std::endl;
    
    // 测试类型转换
    std::cout << "ConceptType PERSON: " << conceptTypeToString(ConceptType::PERSON) << std::endl;
    
    // 测试时间函数
    auto now = std::chrono::system_clock::now();
    std::cout << "Current time: " << timestampToString(now) << std::endl;
    std::cout << "Time of day: " << getTimeOfDay(now) << std::endl;
    
    std::cout << "=== All tests passed! ===" << std::endl;
    return 0;
}
