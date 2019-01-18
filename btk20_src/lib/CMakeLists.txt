foreach(file __init__.py utils.py pybeamformer.py pylocalizer.py)
       install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${file} DESTINATION ${PYTHON_SITE_PACKAGES})
endforeach()
