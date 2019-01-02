include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_tde tde.cc)
target_link_libraries(btk20_tde
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_feature btk20_modulated)

set_source_files_properties(tde.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(tde.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(tde.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(tde python tde.i)
else()
   swig_add_library(tde
        LANGUAGE python
        SOURCES tde.i)
endif()

swig_link_libraries(tde btk20_tde ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/tde.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_tde
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_tde_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/tde/tde.py DESTINATION ${PYTHON_SITE_PACKAGES})
