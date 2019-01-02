include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_localization localization.cc mcc_localizer.cc)
target_link_libraries(btk20_localization
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_feature btk20_square_root)

set_source_files_properties(localization.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(localization.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(localization.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(localization python localization.i)
else()
   swig_add_library(localization
        LANGUAGE python
        SOURCES localization.i)
endif()

swig_link_libraries(localization btk20_localization ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/localization.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_localization
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_localization_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/localization/localization.py DESTINATION ${PYTHON_SITE_PACKAGES})
