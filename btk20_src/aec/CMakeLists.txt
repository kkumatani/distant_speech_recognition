include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_aec aec.cc)
target_link_libraries(btk20_aec
        GSL::gsl GSL::gslcblas
        btk20_stream)

set_source_files_properties(aec.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(aec.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(aec.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(aec python aec.i)
else()
   swig_add_library(aec
        LANGUAGE python
        SOURCES aec.i)
endif()
swig_link_libraries(aec btk20_aec ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/aec.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_aec
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_aec_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/aec/aec.py DESTINATION ${PYTHON_SITE_PACKAGES})
