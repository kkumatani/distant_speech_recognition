include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_dereverberation dereverberation.cc)
target_link_libraries(btk20_dereverberation
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_matrix btk20_feature)

set_source_files_properties(dereverberation.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(dereverberation.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(dereverberation.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(dereverberation python dereverberation.i)
else()
   swig_add_library(dereverberation
        LANGUAGE python
        SOURCES dereverberation.i)
endif()

swig_link_libraries(dereverberation btk20_dereverberation ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/dereverberation.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_dereverberation
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_dereverberation_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/dereverberation/dereverberation.py DESTINATION ${PYTHON_SITE_PACKAGES})
