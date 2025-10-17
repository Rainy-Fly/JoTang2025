cmake_minimum_required(VERSION 3.10)

project(cxxmoon)

find_package(Eigen3 REQUIRED)

###
#add_executable(moon process_moon.cpp)


#target_include_directories(moon PRIVATE /home/fushao/autodiff)

#target_link_libraries(moon Eigen3::Eigen )

###
add_executable(demo demo.cpp)


target_include_directories(demo PRIVATE /home/fushao/autodiff)

target_link_libraries(demo Eigen3::Eigen )