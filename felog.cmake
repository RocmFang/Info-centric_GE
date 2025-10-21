function(add_app_exec EXEC_NAME)
	#link_directories(/opt/intel/oneapi/mkl/2022.0.2/lib/intel64 )
    add_executable(${EXEC_NAME} ${EXEC_NAME}.cpp)
    #target_link_libraries(${EXEC_NAME} PUBLIC ${MPI_LIBRARIES}  -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl)
    target_link_libraries(${EXEC_NAME} PUBLIC ${MPI_LIBRARIES} )
endfunction(add_app_exec)


function(add_cuda_exec EXEC_NAME)
    include_directories(/usr/local/cuda-11.6/targets/x86_64-linux/include/)
    link_directories(/usr/local/cuda-11.6/targets/x86_64-linux/lib/)
    cuda_add_executable(${EXEC_NAME} ${EXEC_NAME}.cpp ${CMAKE_SOURCE_DIR}/src/word2vec.cu )
    target_link_libraries(${EXEC_NAME} ${MPI_LIBRARIES} )
endfunction(add_cuda_exec EXEC_NAME)


function(add_tool_exec EXEC_NAME)
    add_executable(${EXEC_NAME} ${EXEC_NAME}.cpp)
    target_link_libraries(${EXEC_NAME} PUBLIC ${MPI_LIBRARIES})
endfunction(add_tool_exec)
