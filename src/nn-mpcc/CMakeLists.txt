cmake_minimum_required(VERSION 3.7)
project(nn-mpcc)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-O2")

find_package(Python COMPONENTS Development)
find_package(Torch REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)
find_package(
  catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
	std_msgs
	message_generation
)

include_directories(.)
include_directories(External/blasfeo/lib/include)
include_directories(External/hpipm/lib/include)
include_directories(External/matplotlib)
include_directories(External/Eigen)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories(
  ${Python_INCLUDE_DIRS}
)
catkin_package(
)

add_executable(mpc
  main.cpp
  types.cpp
  types.h
  config.h
  Params/params.cpp
  Spline/cubic_spline.cpp
  Spline/arc_length_spline.cpp
  Interfaces/hpipm_interface.cpp
  Interfaces/solver_interface.cpp
  Constraints/constraints.cpp
  Constraints/bounds.cpp
  Cost/cost.cpp
  Model/model.cpp
  Model/nn.cpp
  Model/integrator.cpp
  Tests/spline_test.cpp
  Tests/cost_test.cpp
  Tests/model_integrator_test.cpp
  Tests/constratins_test.cpp
  MPC/mpc.cpp
  Params/track.cpp
  Params/track.h
  Plotting/plotting.cpp
  Plotting/plotting.h
)
add_executable(file_test file_test)
add_executable(random_walker random_walker)
add_executable(bag_process bag_process)

add_library(blasfeo STATIC IMPORTED)
set_property(TARGET blasfeo PROPERTY IMPORTED_LOCATION /home/alexzheng/Documents/GitHub/real-NN-MPCC/src/nn-mpcc/External/blasfeo/lib/lib/libblasfeo.a)
add_library(hpipm STATIC IMPORTED)
set_property(TARGET hpipm PROPERTY IMPORTED_LOCATION /home/alexzheng/Documents/GitHub/real-NN-MPCC/src/nn-mpcc/External/hpipm/lib/lib/libhpipm.a)
add_library(nlohmann_json STATIC IMPORTED)

link_directories(External/hpipm/lib External/blasfeo/lib)
target_include_directories(mpc PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(mpc ${catkin_LIBRARIES} ${Python_LIBRARIES} ${TORCH_LIBRARIES} hpipm blasfeo nlohmann_json::nlohmann_json)
target_link_libraries(random_walker ${catkin_LIBRARIES})
target_link_libraries(bag_process ${catkin_LIBRARIES})
 
