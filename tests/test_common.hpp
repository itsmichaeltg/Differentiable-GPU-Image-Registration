#pragma once

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#define CHECK_TRUE(condition)                                                                    \
    do {                                                                                         \
        if (!(condition)) {                                                                       \
            std::cerr << "CHECK_TRUE failed at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << #condition << std::endl;                                                \
            std::exit(1);                                                                         \
        }                                                                                        \
    } while (0)

#define CHECK_EQ(actual, expected)                                                               \
    do {                                                                                         \
        const auto actual_value = (actual);                                                       \
        const auto expected_value = (expected);                                                   \
        if (!(actual_value == expected_value)) {                                                  \
            std::cerr << "CHECK_EQ failed at " << __FILE__ << ":" << __LINE__ << ": "         \
                      << #actual << " = " << actual_value << ", " << #expected                 \
                      << " = " << expected_value << std::endl;                                  \
            std::exit(1);                                                                         \
        }                                                                                        \
    } while (0)

#define CHECK_NEAR(actual, expected, tolerance)                                                  \
    do {                                                                                         \
        const double actual_value = static_cast<double>(actual);                                  \
        const double expected_value = static_cast<double>(expected);                              \
        const double tolerance_value = static_cast<double>(tolerance);                            \
        if (std::fabs(actual_value - expected_value) > tolerance_value) {                         \
            std::cerr << "CHECK_NEAR failed at " << __FILE__ << ":" << __LINE__ << ": "       \
                      << #actual << " = " << actual_value << ", " << #expected                 \
                      << " = " << expected_value << ", tolerance = " << tolerance_value         \
                      << std::endl;                                                              \
            std::exit(1);                                                                         \
        }                                                                                        \
    } while (0)

#define CHECK_THROWS(expression)                                                                 \
    do {                                                                                         \
        bool threw = false;                                                                       \
        try {                                                                                     \
            (void)(expression);                                                                   \
        } catch (const std::exception&) {                                                         \
            threw = true;                                                                         \
        }                                                                                         \
        if (!threw) {                                                                             \
            std::cerr << "CHECK_THROWS failed at " << __FILE__ << ":" << __LINE__ << ": "     \
                      << #expression << std::endl;                                               \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)
