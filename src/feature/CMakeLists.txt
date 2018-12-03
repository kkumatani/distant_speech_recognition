include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_feature feature.cc lpc.cc spectralestimator.cc videofeature.cc)
target_link_libraries(btk20_feature
        GSL::gsl GSL::gslcblas ${SNDFILE_LIBRARY}
        btk20_common btk20_stream btk20_matrix)

set_source_files_properties(feature.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(feature.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(feature.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(feature python feature.i)
else()
   swig_add_library(feature
        LANGUAGE python
        SOURCES feature.i)
endif()

swig_link_libraries(feature btk20_feature ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/feature.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_feature
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_feature_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/feature/feature.py DESTINATION ${PYTHON_SITE_PACKAGES})
