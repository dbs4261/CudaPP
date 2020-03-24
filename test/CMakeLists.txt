macro(package_add_test_with_libraries TESTNAME FILES LIBRARIES TEST_WORKING_DIRECTORY)
    add_executable(${TESTNAME} ${FILES} ${CMAKE_SOURCE_DIR}/test/test_main.cpp)
    target_link_libraries(${TESTNAME} project_options project_warnings Threads::Threads gtest gmock gtest_main ${LIBRARIES} ${CUDA_LIBRARIES})
    gtest_discover_tests(${TESTNAME}
#            WORKING_DIRECTORY ${TEST_WORKING_DIRECTORY}
#            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${TEST_WORKING_DIRECTORY}"
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    set_target_properties(${TESTNAME} PROPERTIES
        CUDA_SEPERABLE_COMPILATION ON
        CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER}
    )
endmacro()

add_subdirectory(cudapp)