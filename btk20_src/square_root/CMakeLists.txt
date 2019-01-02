include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_square_root square_root.cc)
target_link_libraries(btk20_square_root
        GSL::gsl GSL::gslcblas
        btk20_common)

set_source_files_properties(square_root.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(square_root.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(square_root.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(square_root python square_root.i)
else()
   swig_add_library(square_root
        LANGUAGE python
        SOURCES square_root.i)
endif()

swig_link_libraries(square_root btk20_square_root ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/square_root.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_square_root
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_square_root_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/square_root/square_root.py DESTINATION ${PYTHON_SITE_PACKAGES})
