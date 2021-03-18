set(fbs_flags "--no-includes;--gen-compare;--force-empty")

compile_flatbuffers_schema_to_cpp_opt(test0_fbs_header test0.fbs "${fbs_flags}")

add_custom_target(fbs_headers ALL DEPENDS test0_fbs)

file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/.fbs_dummy.cc CONTENT "")
add_library(fbs_headers_dummy STATIC ${CMAKE_CURRENT_BINARY_DIR}/.fbs_dummy.cc)
add_dependencies(fbs_headers_dummy fbs_headers)
link_libraries(fbs_headers_dummy)
