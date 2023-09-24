set(WORK_SPACE_PATH ${PROJECT_SOURCE_DIR})
configure_file (
  ${PROJECT_SOURCE_DIR}/src/common/GlobalDefination.h.in
  ${PROJECT_BINARY_DIR}/include/GlobalDefination.h)
include_directories(${PROJECT_BINARY_DIR}/include)
