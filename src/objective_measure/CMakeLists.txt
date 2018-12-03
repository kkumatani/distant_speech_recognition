include_directories(${GSL_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDES})
add_library(btk20_objective_measure objective_measure.cc)
target_link_libraries(btk20_objective_measure
        GSL::gsl GSL::gslcblas
        btk20_stream btk20_feature btk20_modulated)

set_source_files_properties(objective_measure.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(objective_measure.i PROPERTIES SWIG_FLAGS "-includeall")
set_source_files_properties(objective_measure.i PROPERTIES SWIG_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}")

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
   swig_add_module(objective_measure python objective_measure.i)
else()
   swig_add_library(objective_measure
        LANGUAGE python
        SOURCES objective_measure.i)
endif()

swig_link_libraries(objective_measure btk20_objective_measure ${PYTHON_LIBRARIES})

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/objective_measure.h
              DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS btk20_objective_measure
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS ${SWIG_MODULE_objective_measure_REAL_NAME}
                ARCHIVE DESTINATION ${PYTHON_SITE_PACKAGES}
                LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/objective_measure/objective_measure.py DESTINATION ${PYTHON_SITE_PACKAGES})
