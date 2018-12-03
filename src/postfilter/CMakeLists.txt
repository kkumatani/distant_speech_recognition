include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_postfilter postfilter.cc binauralprocessing.cc
postfilter.cc spectralsubtraction.cc ${CMAKE_SOURCE_DIR}/beamformer/beamformer.cc)
target_link_libraries(btk20_postfilter
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_matrix btk20_modulated)

set_source_files_properties(postfilter.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(postfilter.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(postfilter.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(postfilter python postfilter.i)
else()
   swig_add_library(postfilter
        LANGUAGE python
        SOURCES postfilter.i)
endif()

swig_link_libraries(postfilter btk20_postfilter ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/binauralprocessing.h
              ${CMAKE_CURRENT_SOURCE_DIR}/postfilter.h
              ${CMAKE_CURRENT_SOURCE_DIR}/spectralsubtraction.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_postfilter
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_postfilter_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/postfilter/postfilter.py DESTINATION ${PYTHON_SITE_PACKAGES})
