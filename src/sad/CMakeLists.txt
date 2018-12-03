include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_sad sad.cc ica.cc sad_feature.cc neural_spnsp_det.c)
target_link_libraries(btk20_sad
        GSL::gsl GSL::gslcblas 
        btk20_stream btk20_feature)

set_source_files_properties(sad.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(sad.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(sad.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(sad python sad.i)
else()
   swig_add_library(sad
        LANGUAGE python
        SOURCES sad.i)
endif()

swig_link_libraries(sad btk20_sad ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/sad.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_sad
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_sad_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/sad/sad.py DESTINATION ${PYTHON_SITE_PACKAGES})
