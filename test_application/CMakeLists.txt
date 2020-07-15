add_executable(Test main.cpp)
target_link_libraries(Test cudapp)

add_executable(SMID_Test smid_test.cu)
target_link_libraries(SMID_Test cudapp)