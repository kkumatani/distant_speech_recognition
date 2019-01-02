#  This file is modified based on
#  https://github.com/neXyon/audaspace/blob/master/cmake/FindLibSndFile.cmake
#
#  SNDFILE_FOUND - system has libsndfile
#  SNDFILE_INCLUDE_DIRS - the libsndfile include directories
#  SNDFILE_LIBRARIES - link these to use libsndfile

# find_package(SNDFILE REQUIRED)
# find_library(SNDFILE sndfile )
# MESSAGE(STATUS "Found SNDFILE: ${SNDFILE}")

# Use pkg-config to get hints about paths
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
        pkg_check_modules(SNDFILE_PKGCONF sndfile)
endif(PKG_CONFIG_FOUND)

# Include dir
find_path(SNDFILE_INCLUDE_DIR
        NAMES sndfile.h
        PATHS ${SNDFILE_PKGCONF_INCLUDE_DIRS} REQUIRED
)

# Library
find_library(SNDFILE_LIBRARY
        NAMES sndfile libsndfile-1
        PATHS ${SNDFILE_PKGCONF_LIBRARY_DIRS} REQUIRED
)

find_package(PackageHandleStandardArgs)
find_package_handle_standard_args(SndFile DEFAULT_MSG SNDFILE_LIBRARY SNDFILE_INCLUDE_DIR)

if(SNDFILE_FOUND)
        message(STATUS "SNDFILE_INCLUDE_DIR = ${SNDFILE_INCLUDE_DIR}")
        message(STATUS "SNDFILE_LIBRARY     = ${SNDFILE_LIBRARY}")
        set(SNDFILE_LIBRARIES ${SNDFILE_LIBRARY})
        set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
endif(SNDFILE_FOUND)

mark_as_advanced(SNDFILE_LIBRARY SNDFILE_LIBRARIES SNDFILE_INCLUDE_DIR SNDFILE_INCLUDE_DIRS)
