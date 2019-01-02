include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_matrix gslmatrix.cc blas1_c.cc linpack_c.cc)
target_link_libraries(btk20_matrix GSL::gsl GSL::gslcblas btk20_common)

set_source_files_properties(matrix.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(matrix.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(matrix.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(matrix python matrix.i)
else()
   swig_add_library(matrix
        LANGUAGE python
        SOURCES matrix.i)
endif()
swig_link_libraries(matrix btk20_matrix ${PYTHON_LIBRARIES})

#install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/matrix.h
#              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_matrix
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_matrix_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/matrix/matrix.py DESTINATION ${PYTHON_SITE_PACKAGES})
