#include <iostream>
#include <string.h>
#include <string>

// v16si 表示 16 个 int 的向量（数组），长度为 64 字节
typedef int v16si __attribute__ ((vector_size (64)));

std::string toString(const v16si & v) {
    std::string s = "[ ";
    for (int i = 0; i < 16; i++) {
        s += std::to_string(v[i]) + " ";
    }
    s += "]";
    return s;
}

int main() {
    v16si v0, v1;
    memset(&v0, 0, sizeof(v0));
    memset(&v1, 0, sizeof(v1));
    std::cout << "v0: " << toString(v0) << std::endl;
    std::cout << "v1: " << toString(v1) << std::endl;

    v0 = v0 + 1;  // v0 + {1, 1, ..., 1}
    std::cout << "v0 = v0 + 1: " << toString(v0) << std::endl;

    v1 = v1 + 2;  // v1 + {2, 3, ..., 2}
    std::cout << "v1 = v1 + 2: " << toString(v1) << std::endl;

    v0 = v0 + v1;
    std::cout << "v0 = v0 + v1: " << toString(v0) << std::endl;

    v1 = v0 * v1;
    std::cout << "v1 = v0 * v1: " << toString(v1) << std::endl;

    v16si a = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    v16si b = {0, 2, 4, 6, 8, 1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15};
    // Vectors are compared element-wise producing 0 when comparison is false 
    // and -1 (constant of the appropriate type where all bits are set) otherwise. 
    auto c = a > b;
    std::cout << "c = a > b: " << toString(c) << std::endl;

    auto d = (a > b) ? v0 : v1;
    std::cout << "d = (a > b) ? v0 : v1: " << toString(d) << std::endl;

    return 0;
}