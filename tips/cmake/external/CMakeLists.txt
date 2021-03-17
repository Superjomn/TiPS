INCLUDE(ExternalProject)

# Introduce variables:
# * CMAKE_INSTALL_LIBDIR
INCLUDE(GNUInstallDirs)
SET(LIBDIR "lib")
if(CMAKE_INSTALL_LIBDIR MATCHES ".*lib64$")
    SET(LIBDIR "lib64")
endif()

SET(FLATBUFFERS_PREFIX_DIR ${THIRD_PARTY_PATH}/flatbuffers/prefix)
SET(FLATBUFFERS_SOURCES_DIR ${THIRD_PARTY_PATH}/flatbuffers/source_code)
SET(FLATBUFFERS_INSTALL_DIR ${THIRD_PARTY_PATH}/install/flatbuffers)
SET(FLATBUFFERS_INCLUDE_DIR "${FLATBUFFERS_SOURCES_DIR}/include" CACHE PATH "flatbuffers include directory." FORCE)
IF(WIN32)
    set(FLATBUFFERS_LIBRARIES "${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/flatbuffers.lib" CACHE FILEPATH "FLATBUFFERS_LIBRARIES" FORCE)
ELSE(WIN32)
    set(FLATBUFFERS_LIBRARIES "${FLATBUFFERS_INSTALL_DIR}/${LIBDIR}/libflatbuffers.a" CACHE FILEPATH "FLATBUFFERS_LIBRARIES" FORCE)
ENDIF(WIN32)

INCLUDE_DIRECTORIES(${FLATBUFFERS_INCLUDE_DIR})

if(NOT HOST_CXX_COMPILER)
    set(HOST_CXX_COMPILER ${CMAKE_CXX_COMPILER})
    set(HOST_C_COMPILER ${CMAKE_C_COMPILER})
endif()

SET(OPTIONAL_ARGS "-DCMAKE_CXX_COMPILER=${HOST_CXX_COMPILER}"
        "-DCMAKE_C_COMPILER=${HOST_C_COMPILER}")

ExternalProject_Add(
        extern_flatbuffers
        ${EXTERNAL_PROJECT_LOG_ARGS}
        #    GIT_REPOSITORY  "https://github.com/google/flatbuffers.git"
        URL             https://paddlelite-data.bj.bcebos.com/third_party_libs/flatbuffers-1.12.0.zip
        GIT_TAG         "v1.12.0"
        SOURCE_DIR      ${FLATBUFFERS_SOURCES_DIR}
        PREFIX          ${FLATBUFFERS_PREFIX_DIR}
        UPDATE_COMMAND  ""
        CMAKE_ARGS      -DBUILD_STATIC_LIBS=ON
        -DCMAKE_INSTALL_PREFIX=${FLATBUFFERS_INSTALL_DIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_TESTING=OFF
        -DCMAKE_BUILD_TYPE=${THIRD_PARTY_BUILD_TYPE}
        -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}
        -DFLATBUFFERS_BUILD_TESTS=OFF
        ${CROSS_COMPILE_CMAKE_ARGS}
        ${OPTIONAL_ARGS}
        ${EXTERNAL_OPTIONAL_ARGS}
        CMAKE_CACHE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=${FLATBUFFERS_INSTALL_DIR}
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=${THIRD_PARTY_BUILD_TYPE}
)
ADD_LIBRARY(flatbuffers STATIC IMPORTED GLOBAL)
SET_PROPERTY(TARGET flatbuffers PROPERTY IMPORTED_LOCATION ${FLATBUFFERS_LIBRARIES})
ADD_DEPENDENCIES(flatbuffers extern_flatbuffers)

SET(FLATBUFFERS_FLATC_EXECUTABLE ${FLATBUFFERS_INSTALL_DIR}/bin/flatc)

include_directories(${FLATBUFFERS_INCLUDE_DIR})

function(register_generated_output file_name)
    get_property(tmp GLOBAL PROPERTY FBS_GENERATED_OUTPUTS)
    list(APPEND tmp ${file_name})
    set_property(GLOBAL PROPERTY FBS_GENERATED_OUTPUTS ${tmp})
endfunction(register_generated_output)

function(compile_flatbuffers_schema_to_cpp_opt TARGET SRC_FBS OPT)
    if(FLATBUFFERS_BUILD_LEGACY)
        set(OPT ${OPT};--cpp-std c++0x)
    else()
        # --cpp-std is defined by flatc default settings.
    endif()
    message(STATUS "`${SRC_FBS}`: add generation of C++ code with '${OPT}'")
    get_filename_component(SRC_FBS_DIR ${SRC_FBS} PATH)
    message(STATUS "SRC_FBS_DIR: ${SRC_FBS_DIR}")
    string(REGEX REPLACE "\\.fbs$" "_generated.h" GEN_HEADER ${SRC_FBS})
    add_custom_command(
            OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${GEN_HEADER}"
            COMMAND "${FLATBUFFERS_FLATC_EXECUTABLE}"
            --cpp --gen-mutable --gen-object-api --reflect-names
            ${OPT}
            -o "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS}"
            DEPENDS flatbuffers ${SRC_FBS}
            COMMENT "Run generation: '${GEN_HEADER}'")
    register_generated_output(${GEN_HEADER})
    add_custom_target(${TARGET} ALL DEPENDS ${GEN_HEADER})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/${SRC_FBS_DIR})
endfunction()
