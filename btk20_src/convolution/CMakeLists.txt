include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_convolution convolution.cc)
target_link_libraries(btk20_convolution
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_feature)

set_source_files_properties(convolution.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(convolution.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(convolution.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(convolution python convolution.i)
else()
   swig_add_library(convolution
        LANGUAGE python
        SOURCES convolution.i)
endif()

swig_link_libraries(convolution btk20_convolution ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/convolution.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_convolution
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_convolution_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/convolution/convolution.py DESTINATION ${PYTHON_SITE_PACKAGES})
