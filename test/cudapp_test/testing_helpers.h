//
// Created by developer on 3/23/20.
//

#ifndef CUDAPP_TESTING_HELPERS_H
#define CUDAPP_TESTING_HELPERS_H

#include <random>
#include <type_traits>
#include <vector>

namespace cudapp {
namespace test {

/**
 * @breif A typedef for generating either floating point or integer random numbers with one template.
 */
template <typename T>
using uniform_distribution = typename std::conditional<std::is_floating_point<T>::value,
  typename std::enable_if<std::is_arithmetic<T>::value, std::uniform_real_distribution<T>>::type,
  typename std::enable_if<std::is_arithmetic<T>::value, std::uniform_int_distribution<T>>::type>::type;

/**
 * @breif Creates a single value randomly picked from a uniform distibution between minimum and maximum.
 * @tparam T The type of the value to be generated.
 * @param minimum The minimum possible value [std::numeric_limits<T>::min()]
 * @param maximum The maximum possible value [std::numeric_limits<T>::max()]
 * @return The random value generated
 */
template <typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
T CreateUniformRandom(T minimum=std::numeric_limits<T>::min(), T maximum=std::numeric_limits<T>::max()) {
  std::random_device device;
  std::mt19937 generator(device());
  uniform_distribution<T> distribution(minimum, maximum);
  return distribution(generator);
}

/**
 * @brief
 * @tparam T
 * @tparam Allocator
 * @param size
 * @param minimum
 * @param maximum
 * @return
 */
template <typename T, typename Allocator=std::allocator<T>, typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
std::vector<T, Allocator> GenerateUniformlyRandom(unsigned int size, T minimum=std::numeric_limits<T>::min(), T maximum=std::numeric_limits<T>::max()) {
  std::random_device device;
  std::mt19937 generator(device());
  uniform_distribution<T> distribution(minimum, maximum);
  std::vector<T, Allocator> out(size);
  std::generate(out.begin(), out.end(), [&generator, &distribution]()->T{return distribution(generator);});
  return out;
}

}
}

#endif//CUDAPP_TESTING_HELPERS_H
