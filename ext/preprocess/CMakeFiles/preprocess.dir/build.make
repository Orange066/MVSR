# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess

# Include any dependencies generated for this target.
include CMakeFiles/preprocess.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/preprocess.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/preprocess.dir/flags.make

CMakeFiles/preprocess.dir/main.cpp.o: CMakeFiles/preprocess.dir/flags.make
CMakeFiles/preprocess.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/preprocess.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/preprocess.dir/main.cpp.o -c /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/main.cpp

CMakeFiles/preprocess.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/preprocess.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/main.cpp > CMakeFiles/preprocess.dir/main.cpp.i

CMakeFiles/preprocess.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/preprocess.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/main.cpp -o CMakeFiles/preprocess.dir/main.cpp.s

CMakeFiles/preprocess.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/preprocess.dir/main.cpp.o.requires

CMakeFiles/preprocess.dir/main.cpp.o.provides: CMakeFiles/preprocess.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/preprocess.dir/build.make CMakeFiles/preprocess.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/preprocess.dir/main.cpp.o.provides

CMakeFiles/preprocess.dir/main.cpp.o.provides.build: CMakeFiles/preprocess.dir/main.cpp.o


CMakeFiles/preprocess.dir/preprocess.cpp.o: CMakeFiles/preprocess.dir/flags.make
CMakeFiles/preprocess.dir/preprocess.cpp.o: preprocess.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/preprocess.dir/preprocess.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/preprocess.dir/preprocess.cpp.o -c /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/preprocess.cpp

CMakeFiles/preprocess.dir/preprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/preprocess.dir/preprocess.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/preprocess.cpp > CMakeFiles/preprocess.dir/preprocess.cpp.i

CMakeFiles/preprocess.dir/preprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/preprocess.dir/preprocess.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/preprocess.cpp -o CMakeFiles/preprocess.dir/preprocess.cpp.s

CMakeFiles/preprocess.dir/preprocess.cpp.o.requires:

.PHONY : CMakeFiles/preprocess.dir/preprocess.cpp.o.requires

CMakeFiles/preprocess.dir/preprocess.cpp.o.provides: CMakeFiles/preprocess.dir/preprocess.cpp.o.requires
	$(MAKE) -f CMakeFiles/preprocess.dir/build.make CMakeFiles/preprocess.dir/preprocess.cpp.o.provides.build
.PHONY : CMakeFiles/preprocess.dir/preprocess.cpp.o.provides

CMakeFiles/preprocess.dir/preprocess.cpp.o.provides.build: CMakeFiles/preprocess.dir/preprocess.cpp.o


# Object files for target preprocess
preprocess_OBJECTS = \
"CMakeFiles/preprocess.dir/main.cpp.o" \
"CMakeFiles/preprocess.dir/preprocess.cpp.o"

# External object files for target preprocess
preprocess_EXTERNAL_OBJECTS =

preprocess.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/preprocess.dir/main.cpp.o
preprocess.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/preprocess.dir/preprocess.cpp.o
preprocess.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/preprocess.dir/build.make
preprocess.cpython-37m-x86_64-linux-gnu.so: CMakeFiles/preprocess.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared module preprocess.cpython-37m-x86_64-linux-gnu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/preprocess.dir/link.txt --verbose=$(VERBOSE)
	/usr/bin/strip /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/preprocess.cpython-37m-x86_64-linux-gnu.so

# Rule to build all files generated by this target.
CMakeFiles/preprocess.dir/build: preprocess.cpython-37m-x86_64-linux-gnu.so

.PHONY : CMakeFiles/preprocess.dir/build

CMakeFiles/preprocess.dir/requires: CMakeFiles/preprocess.dir/main.cpp.o.requires
CMakeFiles/preprocess.dir/requires: CMakeFiles/preprocess.dir/preprocess.cpp.o.requires

.PHONY : CMakeFiles/preprocess.dir/requires

CMakeFiles/preprocess.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/preprocess.dir/cmake_clean.cmake
.PHONY : CMakeFiles/preprocess.dir/clean

CMakeFiles/preprocess.dir/depend:
	cd /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess /opt/disk1/chengri/MVSRIP_10_11_dpesti_syq/ext/preprocess/CMakeFiles/preprocess.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/preprocess.dir/depend

