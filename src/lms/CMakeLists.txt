include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_lms lms.cc)
target_link_libraries(btk20_lms
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_matrix btk20_convolution)

set_source_files_properties(lms.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(lms.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(lms.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(lms python lms.i)
else()
   swig_add_library(lms
        LANGUAGE python
        SOURCES lms.i)
endif()

swig_link_libraries(lms btk20_lms ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lms.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_lms
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_lms_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/lms/lms.py DESTINATION ${PYTHON_SITE_PACKAGES})
