add_library(btk20_utils utils.cc)

set_source_files_properties(utils.i PROPERTIES CPLUSPLUS ON)
set_source_files_properties(utils.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(utils.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(utils python utils.i utils.cc)
else()
   swig_add_library(utils
        LANGUAGE python
        SOURCES utils.i utils.cc)
endif()
swig_link_libraries(utils btk20_utils ${PYTHON_LIBRARIES})

install(FILES utils.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_utils
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_utils_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
#install(FILES utils.py DESTINATION ${PYTHON_SITE_PACKAGES})
