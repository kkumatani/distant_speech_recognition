include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_stream stream.cc file_stream.cc)
target_link_libraries(btk20_stream GSL::gsl GSL::gslcblas btk20_common)

set_source_files_properties(stream.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(stream.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(stream.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(stream python stream.i stream.cc)
else()
   swig_add_library(stream
        LANGUAGE python
        SOURCES stream.i stream.cc)
endif()
swig_link_libraries(stream btk20_stream ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/stream.h
              ${CMAKE_CURRENT_SOURCE_DIR}/pyStream.h
              ${CMAKE_CURRENT_SOURCE_DIR}/file_stream.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_stream
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_stream_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/stream/stream.py DESTINATION ${PYTHON_SITE_PACKAGES})
