llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

set(LLAMA_CORE_SOURCES
    llama.cpp
    llama-adapter.cpp
    llama-arch.cpp
    llama-batch.cpp
    llama-chat.cpp
    llama-context.cpp
    llama-cparams.cpp
    llama-grammar.cpp
    llama-graph.cpp
    llama-hparams.cpp
    llama-impl.cpp
    llama-io.cpp
    llama-kv-cache.cpp
    llama-kv-cache-iswa.cpp
    llama-kv-cache-dsa.cpp
    llama-kv-cache-dsa-iswa.cpp
    llama-kv-cache-msa.cpp
    llama-kv-cache-dsv4.cpp
    llama-memory.cpp
    llama-moe-cache.cpp
    llama-memory-hybrid.cpp
    llama-memory-hybrid-iswa.cpp
    llama-memory-hybrid-idx.cpp
    llama-memory-recurrent.cpp
    llama-backing-store.cpp
    llama-codec.cpp
    llama-codec-xz.cpp
    llama-mmap.cpp
    llama-model-loader.cpp
    llama-model-saver.cpp
    llama-model-source.cpp
    llama-model.cpp
    llama-quant.cpp
    llama-sampler.cpp
    llama-vocab.cpp
    unicode-data.cpp
    unicode.cpp
)

add_library(llama
            ../include/llama.h
            ${LLAMA_CORE_SOURCES}
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_VERSION_BASE}
    SOVERSION ${LLAMA_VERSION_MAJOR}
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
    UNITY_BUILD ON
    UNITY_BUILD_BATCH_SIZE 16
)

# exclude non-model sources from unity build
set_source_files_properties(${LLAMA_CORE_SOURCES} ../include/llama.h unicode.h
    PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

configure_file(llama-version.h.in ${CMAKE_CURRENT_BINARY_DIR}/llama-version.h @ONLY)

target_include_directories(llama PRIVATE . ${CMAKE_CURRENT_BINARY_DIR})
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# optional codecs for compressed models. Without one, a file in that format is refused with an explicit error
# <option> is AUTO (use the library if found), ON (require it) or OFF, the result is <option>_ENABLED
function(llama_optional_codec option package found_var target define label)
    set(enabled OFF)
    string(TOUPPER "${${option}}" mode)
    if (NOT mode MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR "${option} must be AUTO, ON or OFF, got '${${option}}'")
    endif()
    if (NOT mode STREQUAL "OFF")
        if (mode STREQUAL "ON")
            find_package(${package} REQUIRED)
        else()
            find_package(${package})
        endif()
        if (${found_var})
            set(enabled ON)
            target_link_libraries(llama PRIVATE ${target})
            target_compile_definitions(llama PRIVATE ${define})
        endif()
    endif()
    message(STATUS "llama: ${label} support for compressed models: ${enabled}")
    set(${option}_ENABLED ${enabled} PARENT_SCOPE)
endfunction()

llama_optional_codec(LLAMA_LZMA LibLZMA LIBLZMA_FOUND LibLZMA::LibLZMA LLAMA_USE_LZMA xz)
set(LLAMA_LZMA_ENABLED ${LLAMA_LZMA_ENABLED} PARENT_SCOPE)

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
