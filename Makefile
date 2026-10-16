# Build the borg engine: shared library (for Python) and the `borg` CLI.
#
#   make                 optimized for this machine (-march=native)
#   make PORTABLE=1      generic build for distribution (baseline ISA, scalar kernels on x86)
#   make CC=clang        any C11 compiler works
#   make THREADS=0       single-threaded build (no pthreads), e.g. for WebAssembly

CC ?= cc
PORTABLE ?= 0
THREADS ?= 1
BUILD ?= build

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

CFLAGS ?= -O3
CFLAGS += -std=c11 -Wall -Wextra -Wno-unused-parameter -fPIC -fvisibility=hidden -D_GNU_SOURCE -D_DEFAULT_SOURCE
LDLIBS += -lm

ifeq ($(PORTABLE),0)
  ifneq ($(filter arm64 aarch64,$(UNAME_M)),)
    CFLAGS += -mcpu=native
  else
    CFLAGS += -march=native
  endif
endif

ifeq ($(THREADS),1)
  CFLAGS += -pthread
  LDLIBS += -pthread
else
  CFLAGS += -DBORG_NO_THREADS
endif

ifeq ($(UNAME_S),Darwin)
  SHLIB := $(BUILD)/libborg.dylib
  SHFLAGS := -dynamiclib
else
  SHLIB := $(BUILD)/libborg.so
  SHFLAGS := -shared
endif

LIB_SRC := engine/json.c engine/model.c engine/kernels.c engine/pool.c engine/tokenizer.c \
           engine/forward.c engine/sample.c engine/generate.c engine/json_mode.c
LIB_OBJ := $(patsubst engine/%.c,$(BUILD)/%.o,$(LIB_SRC))
HDRS := $(wildcard engine/*.h)

all: $(SHLIB) $(BUILD)/borg

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: engine/%.c $(HDRS) | $(BUILD)
	$(CC) $(CFLAGS) -c $< -o $@

$(SHLIB): $(LIB_OBJ)
	$(CC) $(SHFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/borg: engine/main.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/test_kernels: tests/c/test_kernels.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -Iengine -o $@ $^ $(LDLIBS)

$(BUILD)/test_json_mode: tests/c/test_json_mode.c $(LIB_OBJ)
	$(CC) $(CFLAGS) -Iengine -o $@ $^ $(LDLIBS)

test-c: $(BUILD)/test_kernels $(BUILD)/test_json_mode
	$(BUILD)/test_kernels
	$(BUILD)/test_json_mode

clean:
	rm -rf $(BUILD)

.PHONY: all clean test-c
