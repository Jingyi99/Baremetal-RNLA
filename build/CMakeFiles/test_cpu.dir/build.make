# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.28.2/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.28.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/xiaowenyuan/Baremetal-RNLA

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/xiaowenyuan/Baremetal-RNLA/build

# Include any dependencies generated for this target.
include CMakeFiles/test_cpu.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test_cpu.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test_cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test_cpu.dir/flags.make

CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o: CMakeFiles/test_cpu.dir/flags.make
CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o: /Users/xiaowenyuan/Baremetal-RNLA/test/cpu/test_cpu.c
CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o: CMakeFiles/test_cpu.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/xiaowenyuan/Baremetal-RNLA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o"
	riscv64-unknown-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o -MF CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o.d -o CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o -c /Users/xiaowenyuan/Baremetal-RNLA/test/cpu/test_cpu.c

CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.i"
	riscv64-unknown-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/xiaowenyuan/Baremetal-RNLA/test/cpu/test_cpu.c > CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.i

CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.s"
	riscv64-unknown-elf-gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/xiaowenyuan/Baremetal-RNLA/test/cpu/test_cpu.c -o CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.s

# Object files for target test_cpu
test_cpu_OBJECTS = \
"CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o"

# External object files for target test_cpu
test_cpu_EXTERNAL_OBJECTS =

test_cpu: CMakeFiles/test_cpu.dir/test/cpu/test_cpu.c.o
test_cpu: CMakeFiles/test_cpu.dir/build.make
test_cpu: libbmrnla.a
test_cpu: CMakeFiles/test_cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/xiaowenyuan/Baremetal-RNLA/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable test_cpu"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test_cpu.dir/build: test_cpu
.PHONY : CMakeFiles/test_cpu.dir/build

CMakeFiles/test_cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/test_cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/test_cpu.dir/clean

CMakeFiles/test_cpu.dir/depend:
	cd /Users/xiaowenyuan/Baremetal-RNLA/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/xiaowenyuan/Baremetal-RNLA /Users/xiaowenyuan/Baremetal-RNLA /Users/xiaowenyuan/Baremetal-RNLA/build /Users/xiaowenyuan/Baremetal-RNLA/build /Users/xiaowenyuan/Baremetal-RNLA/build/CMakeFiles/test_cpu.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/test_cpu.dir/depend

