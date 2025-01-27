NVCC = nvcc
NVCC_FLAGS = -O3 -I./include

SRC_DIR = src
BUILD_DIR = build
TARGET = mnist_filter

SRCS = $(wildcard $(SRC_DIR)/*.cu)
OBJS = $(SRCS:$(SRC_DIR)/%.cu=$(BUILD_DIR)/%.o)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)