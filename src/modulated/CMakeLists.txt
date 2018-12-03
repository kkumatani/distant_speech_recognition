include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_modulated modulated.cc prototype_design.cc pc_lattice.c)
target_link_libraries(btk20_modulated
        GSL::gsl GSL::gslcblas ${SNDFILE_LIBRARY}
        btk20_common btk20_stream)

set_source_files_properties(modulated.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(modulated.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(modulated.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(modulated python modulated.i)
else()
   swig_add_library(modulated
        LANGUAGE python
        SOURCES modulated.i)
endif()

swig_link_libraries(modulated btk20_modulated ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/modulated.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_modulated
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_modulated_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/modulated/modulated.py DESTINATION ${PYTHON_SITE_PACKAGES})
