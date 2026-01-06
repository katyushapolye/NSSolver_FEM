############################
# Compiler
############################
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fopenmp -MMD -MP

############################
# deal.II paths
############################
DEALII_DIR := include/dealii/build
DEALII_INCLUDE := -I$(DEALII_DIR)/include \
                  -I$(DEALII_DIR)/include/deal.II/bundled
DEALII_LIB := -L$(DEALII_DIR)/lib -Wl,-rpath,$(DEALII_DIR)/lib
DEALII_LIBS := -ldeal_II -ltbb -lgomp

############################
# SuiteSparse
############################
SUITESPARSE_INCLUDE := -I/usr/include/suitesparse
SUITESPARSE_LIBS := -lumfpack -lamd -lcholmod -lcolamd -lsuitesparseconfig

############################
# ImGui / ImPlot / ImPlot3D
############################
IMGUI_DIR := include/imgui-1.92.4
IMPLOT_DIR := include/implot-0.17
IMPLOT3D_DIR := include/implot3d-0.3

IMGUI_INCLUDE := -I$(IMGUI_DIR) -I$(IMGUI_DIR)/backends
IMPLOT_INCLUDE := -I$(IMPLOT_DIR)
IMPLOT3D_INCLUDE := -I$(IMPLOT3D_DIR)

############################
# OpenGL / GLFW
############################
GL_LIBS := -lglfw -lGL

############################
# Project directories
############################
SRC_DIR := src
HEADER_DIR := src/headers
BIN_DIR := bin

############################
# Sources
############################
SRCS := $(wildcard $(SRC_DIR)/*.cpp)

IMGUI_SRCS := \
  $(IMGUI_DIR)/imgui.cpp \
  $(IMGUI_DIR)/imgui_draw.cpp \
  $(IMGUI_DIR)/imgui_tables.cpp \
  $(IMGUI_DIR)/imgui_widgets.cpp \
  $(IMGUI_DIR)/imgui_demo.cpp \
  $(IMGUI_DIR)/backends/imgui_impl_glfw.cpp \
  $(IMGUI_DIR)/backends/imgui_impl_opengl3.cpp

IMPLOT_SRCS := \
  $(IMPLOT_DIR)/implot.cpp \
  $(IMPLOT_DIR)/implot_items.cpp \
  $(IMPLOT_DIR)/implot_demo.cpp

IMPLOT3D_SRCS := \
  $(IMPLOT3D_DIR)/implot3d.cpp \
  $(IMPLOT3D_DIR)/implot3d_items.cpp \
  $(IMPLOT3D_DIR)/implot3d_demo.cpp \
  $(IMPLOT3D_DIR)/implot3d_meshes.cpp

ALL_SRCS := $(SRCS) $(IMGUI_SRCS) $(IMPLOT_SRCS) $(IMPLOT3D_SRCS)

############################
# Objects
############################
OBJS := $(patsubst %.cpp,$(BIN_DIR)/%.o,$(notdir $(ALL_SRCS)))
DEPS := $(OBJS:.o=.d)

############################
# Target
############################
TARGET := $(BIN_DIR)/main

############################
# Includes
############################
INCLUDES := \
  -I$(HEADER_DIR) \
  $(DEALII_INCLUDE) \
  $(SUITESPARSE_INCLUDE) \
  $(IMGUI_INCLUDE) \
  $(IMPLOT_INCLUDE) \
  $(IMPLOT3D_INCLUDE)

############################
# Default target
############################
all: $(TARGET)

############################
# Create bin directory
############################
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

############################
# Compile rules
############################
$(BIN_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: $(IMGUI_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: $(IMGUI_DIR)/backends/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: $(IMPLOT_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BIN_DIR)/%.o: $(IMPLOT3D_DIR)/%.cpp | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

############################
# Link
############################
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) \
	$(DEALII_LIB) $(DEALII_LIBS) \
	$(SUITESPARSE_LIBS) \
	$(GL_LIBS) \
	-lpthread -ldl -lrt \
	-o $@

############################
# Run
############################
run: $(TARGET)
	./$(TARGET)

############################
# Clean
############################
clean:
	rm -rf $(BIN_DIR)

############################
# Info
############################
info:
	@echo "deal.II dir: $(DEALII_DIR)"
	@echo "Sources: $(ALL_SRCS)"
	@echo "Objects: $(OBJS)"

-include $(DEPS)

.PHONY: all clean run info
