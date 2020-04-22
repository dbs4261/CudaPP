//
// Created by Daniel Simon on 4/10/20.
//

#ifndef CUDAPP_TYPE_HELPERS_H
#define CUDAPP_TYPE_HELPERS_H

#include <memory>

namespace cudapp {

/**
 * @brief A helper to prevent the function arguments being used for template argument deduction instead of the function.
 * @tparam T The type.
 */
template<typename T> struct identity {using type = T;};
template<typename T> using identity_t = typename identity<T>::type;

}

#endif //CUDAPP_TYPE_HELPERS_H
