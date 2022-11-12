#ifndef UTILS_H_
#define UTILS_H_

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

class Timer {
public:
  Timer(const std::string &name)
      : timer_name_(name), start_(std::chrono::steady_clock::now()) {}

  ~Timer() {
    auto end = std::chrono::steady_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    std::cout << ">>> " << timer_name_ << ": " << duration.count() << "ms"
              << std::endl;
  }

private:
  std::string timer_name_{};
  std::chrono::time_point<std::chrono::steady_clock> start_{};
};


struct RGBImage {
  int cols, rows, channels;
  unsigned char *data;
};

#endif