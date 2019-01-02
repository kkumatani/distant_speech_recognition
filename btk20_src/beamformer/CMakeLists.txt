include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_beamformer beamformer.cc taylorseries.cc modalbeamformer.cc tracker.cc)
target_link_libraries(btk20_beamformer
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_matrix btk20_feature btk20_modulated btk20_postfilter)

set_source_files_properties(beamformer.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(beamformer.i PROPERTIES SWIG_FLAGS "-includeall -Wsometimes-uninitialized")
set_source_files_properties(beamformer.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(beamformer python beamformer.i)
else()
   swig_add_library(beamformer
        LANGUAGE python
        SOURCES beamformer.i)
endif()

swig_link_libraries(beamformer btk20_beamformer ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/beamformer.h
              ${CMAKE_CURRENT_SOURCE_DIR}/modalbeamformer.h
              ${CMAKE_CURRENT_SOURCE_DIR}/tracker.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_beamformer
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_beamformer_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/beamformer/beamformer.py DESTINATION ${PYTHON_SITE_PACKAGES})
