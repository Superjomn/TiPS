add_subdirectory(mpi)
add_subdirectory(common)
add_subdirectory(ps)
add_subdirectory(message)
add_subdirectory(collective)

core_gather_srcs(SRCS operations.cc)
