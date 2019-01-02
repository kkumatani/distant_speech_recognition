add_library(btk20_common common.cc jexception.cc jpython_error.cc
mach_ind_io.cc memory_manager.cc refcount.cc error.c)

set_source_files_properties(common.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(common.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(common.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(common python common.i common.cc)
else()
   swig_add_library(common
        LANGUAGE python
        SOURCES common.i common.cc)
endif()
swig_link_libraries(common btk20_common ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/common.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_common
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_common_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/common/common.py DESTINATION ${PYTHON_SITE_PACKAGES})
