//
// Created by developer on 3/23/20.
//

#include "gtest/gtest.h"

int GetTwo() {
  return 2;
}

TEST(GetTwoTest, Two) {
  EXPECT_EQ(2, GetTwo());
}

TEST(GetTwoTest, Three) {
  EXPECT_EQ(3, GetTwo());
}