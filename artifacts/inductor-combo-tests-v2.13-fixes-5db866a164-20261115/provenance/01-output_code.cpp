

#include <torch_npu/csrc/inductor/aoti_runtime/model.h>
#include <torch_npu/csrc/inductor/aoti_runtime/device_utils.h>
#include <torch_npu/csrc/inductor/aoti_runtime/utils_npu.h>
#include <torch_npu/csrc/inductor/aoti_torch/generated/c_shim_npu.h>

#include <fstream>
#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <runtime/runtime/rt.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/inductor/aoti_include/npu.h>
// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch_npu/csrc/inductor/aoti_runtime/model_container.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Stores the last error message from a failed AOTI runtime call so that
// callers on the other side of the C ABI boundary can retrieve it via
// AOTInductorGetLastError(). Without this, exception messages (e.g.
// "CUDA error: an illegal memory access was encountered") are lost when
// CONVERT_EXCEPTION_TO_ERROR_CODE catches them and returns an error code.
static thread_local std::string g_aoti_last_error;

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)      \
  try {                                           \
    g_aoti_last_error.clear();                    \
    __VA_ARGS__                                   \
  } catch (const std::exception& e) {             \
    g_aoti_last_error = e.what();                 \
    std::cerr << "Error: " << e.what() << '\n';   \
    return AOTI_RUNTIME_FAILURE;                  \
  } catch (...) {                                 \
    g_aoti_last_error = "Unknown exception";      \
    std::cerr << "Unknown exception occurred.\n"; \
    return AOTI_RUNTIME_FAILURE;                  \
  }                                               \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(false);
  }
  AOTINoGradGuard(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard(AOTINoGradGuard&&) noexcept = delete;
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  AOTINoGradGuard& operator=(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard& operator=(AOTINoGradGuard&&) noexcept = delete;
  bool prev_mode{aoti_torch_grad_mode_is_enabled()};
};

namespace {

std::unordered_map<std::string, AtenTensorHandle> constant_map_from_pairs(
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
    input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  return input_map;
}

// Shared constructor for AOTInductorModelCreate / AOTInductorModelCreateV2.
// `populate(constant_map)` is called between model construction and
// optional embedded-blob loading.
template <typename Populate>
AOTIRuntimeError createModelImpl(
    AOTInductorModelHandle* model_handle,
    bool load_constants_from_blob,
    Populate&& populate) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto constant_array = std::make_shared<
        std::vector<torch::aot_inductor::ConstantHandle>>();
    auto* model = new torch::aot_inductor::AOTInductorModel(
        constant_map,
        constant_array,
        // device_str is hardcoded, as AOTInductorModelCreate is only used
        // for CPU models.
        "cpu",
        "");
    populate(*constant_map);
    if (load_constants_from_blob) {
      model->load_constants();
    }
    *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
  })
}

} // namespace

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "npu",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {

  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}


AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *type = container->constant_type(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
  AOTInductorModelContainerHandle container_handle,
  size_t idx,
  size_t* data_size) {
  auto* container =
    reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
        container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *data_size = container->constant_data_size(idx); })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { const auto ret = container->extract_constants_map(use_inactive);
      for (const auto& pair: ret) {
        constants_map->emplace(pair.first, pair.second);
      }
    })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMapEntries(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry** entries,
    size_t* num_entries,
    bool use_inactive) {
  if (!entries || !num_entries) {
    return AOTI_RUNTIME_FAILURE;
  }
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto& extracted =
        container->extract_constants_map_entries(use_inactive);
    *entries = extracted.empty() ? nullptr : extracted.data();
    *num_entries = extracted.size();
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update, /* user_managed = */ true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  // Build a local unordered_map inside
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
      input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map, use_inactive, validate_full_update, /*user_managed=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpu(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map,
        use_inactive,
        validate_full_update,
        /*user_managed=*/false,
        /*allow_h2d_copy=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpuPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map,
        use_inactive,
        validate_full_update,
        /*user_managed=*/false,
        /*allow_h2d_copy=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(
      container_handle,
      constant_map_handle,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  return AOTInductorModelContainerUpdateConstantBufferPairs(
      container_handle,
      pairs,
      num_pairs,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
}

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->free_inactive_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return createModelImpl(
      model_handle, constant_map_handle == nullptr, [=](auto& constant_map) {
        auto* input_map = reinterpret_cast<
            std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);
        if (input_map) {
          for (const auto& kv : *input_map) {
            constant_map.emplace(kv.first, kv.second);
          }
        }
      });
}

AOTIRuntimeError AOTInductorModelCreateV2(
    AOTInductorModelHandle* model_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  return createModelImpl(
      model_handle, pairs == nullptr || num_pairs == 0, [=](auto& constant_map) {
        if (pairs && num_pairs > 0) {
          constant_map.reserve(num_pairs);
          for (size_t i = 0; i < num_pairs; ++i) {
            constant_map.emplace(pairs[i].name, pairs[i].handle);
          }
        }
      });
}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

// C-ABI-safe variant: uses an array of (name, handle) pairs instead of an
// opaque pointer to std::unordered_map, so the host and DSO can use
// different C++ standard libraries without ABI conflicts.
AOTIRuntimeError AOTInductorModelUpdateConstantsMapV2(
    AOTInductorModelHandle model_handle,
    const AOTInductorConstantMapEntry* pairs,
    int32_t num_pairs) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    constant_map->reserve(num_pairs);
    for (int32_t i = 0; i < num_pairs; ++i) {
      constant_map->emplace(pairs[i].name, pairs[i].handle);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantsBlobSize(
    AOTInductorModelContainerHandle container_handle,
    uint64_t* ret_size) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_size = container->constant_blob_size(); })
}


// Load weights from a single blob in weight_blob_ptr
AOTIRuntimeError AOTInductorModelUpdateConstantsFromBlob(
    AOTInductorModelContainerHandle container_handle,
    const uint8_t* weight_blob_ptr){
    auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      {container->update_constants_from_blob(weight_blob_ptr); })
    }


AOTIRuntimeError AOTInductorGetLastError(
    const char** error_msg) {
  *error_msg = g_aoti_last_error.c_str();
  return AOTI_RUNTIME_SUCCESS;
}

} // extern "C"


namespace {

struct Grid {
    Grid(uint32_t x, uint32_t y, uint32_t z)
      : grid_x(x), grid_y(y), grid_z(z) {}
    uint32_t grid_x;
    uint32_t grid_y;
    uint32_t grid_z;

    bool is_non_zero() {
        return grid_x > 0 && grid_y > 0 && grid_z > 0;
    }
};

}  // anonymous namespace

extern "C" {
    typedef int (* callback)(unsigned int type, void* data, unsigned int len);
    extern int MsprofReportApi(unsigned int  agingFlag, const MsprofApi *api);
    extern unsigned long int  MsprofSysCycleTime();
    extern int MsprofRegisterCallback(unsigned int moduleId, callback handle);
    static unsigned int __MsprofFlagL0  = 0;
    static unsigned int __MsprofFlagL1  = 0;

    int ProfCtrlHandle(unsigned int CtrlType, void* CtrlData, unsigned int DataLen) {
        if ((CtrlData == nullptr) || (DataLen == 0U)) {
            return 1;
        }

        if (CtrlType == 1) {
            MsprofCommandHandle* handle = (MsprofCommandHandle *)(CtrlData);
            if (handle->type >= 6)  // 6 is not used here
                return 1;
            if (handle->type == 1) {  // init - 0  , start - 1
                __MsprofFlagL0 = ((0x00000800ULL & handle->profSwitch) == 0x00000800ULL) ? 1 : 0;
                __MsprofFlagL1 = ((0x00000002ULL & handle->profSwitch) == 0x00000002ULL) ? 1 : 0;
            }
        }
        return 0;
    }
}

static std::unordered_map<std::string, std::pair<aclrtBinHandle, aclrtFuncHandle>> registered_kernels;

static inline bool endsWith(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size()
        && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static inline std::string inferKernelModeFromName(const std::string &nameFuncMode) {
    size_t spacePos = nameFuncMode.find(' ');
    if (spacePos != std::string::npos && spacePos + 1 < nameFuncMode.size()) {
        return nameFuncMode.substr(spacePos + 1);
    }
    if (endsWith(nameFuncMode, "_aiv")) {
        return "aiv";
    }
    if (endsWith(nameFuncMode, "_aic")) {
        return "aic";
    }
    return "";
}

static inline std::string parseKernelFuncName(
        const std::string &nameFuncMode,
        const std::string &kernel_mode) {
    size_t spacePos = nameFuncMode.find(' ');
    if (spacePos != std::string::npos) {
        return nameFuncMode.substr(0, spacePos);
    }
    if (!kernel_mode.empty()) {
        const std::string mode_suffix = "_" + kernel_mode;
        if (endsWith(nameFuncMode, mode_suffix)) {
            return nameFuncMode.substr(0, nameFuncMode.size() - mode_suffix.size());
        }
    }
    return nameFuncMode;
}

static inline void * loadKernelFromData(
        const void *data,
        size_t data_size,
        const std::string &nameFunc,
        const std::string &kernel_mode_str,
        uint32_t sharedMemBytes) {
    aclError aclRet;

    uint32_t magic;
    std::string kernel_mode{kernel_mode_str};
    if (kernel_mode.empty()) {
        kernel_mode = inferKernelModeFromName(nameFunc);
    }
    if (kernel_mode == "aiv") {
        magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
    } else {
        magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;
    }

    aclrtBinaryLoadOption optArr[] = {
        { .type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, .value = { .isLazyLoad = 0 } },
        { .type = ACL_RT_BINARY_LOAD_OPT_MAGIC, .value = { .magic = magic } }
    };
    aclrtBinaryLoadOptions loadOptions = { .options = optArr, .numOpt = 2 };
    aclrtBinHandle binHandle = nullptr;
    aclRet = aclrtBinaryLoadFromData(data, data_size, &loadOptions, &binHandle);
    if (aclRet != ACL_SUCCESS) {
        throw std::runtime_error(std::string("aclrtBinaryLoadFromData failed, 0x") + std::to_string(aclRet));
    }

    std::string kernel_func_name = parseKernelFuncName(nameFunc, kernel_mode);
    const char* name = kernel_func_name.c_str();
    aclrtFuncHandle funcHandle = nullptr;
    aclRet = aclrtBinaryGetFunction(binHandle, name, &funcHandle);
    if (aclRet != ACL_SUCCESS) {
        throw std::runtime_error(std::string("aclrtBinaryGetFunction failed(name = ") + name
                    + std::string("), 0x") + std::to_string(aclRet));
    }

    registered_kernels[nameFunc] = std::make_pair(binHandle, funcHandle);

    return reinterpret_cast<void *>(funcHandle);
}

static inline void * loadKernel(
        std::string filePath,
        const std::string &&nameFunc,
        const std::string &&kernel_mode_str,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    if (cubinDir) {
        std::filesystem::path p1{*cubinDir};
        std::filesystem::path p2{filePath};
        filePath = (p1 / p2.filename()).string();
    }

    std::ifstream file(std::string(filePath), std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("open npubin failed"));
    }

    std::streamsize data_size = file.tellg();

    file.seekg(0, std::ios::beg);
    auto buffer = std::make_unique<char[]>(data_size);
    if (!file.read(buffer.get(), data_size)) {
        throw std::runtime_error(std::string("read npubin failed"));
    }

    return loadKernelFromData(buffer.get(), static_cast<size_t>(data_size),
                              nameFunc, kernel_mode_str, sharedMemBytes);
}

static inline void * loadKernel(
        const unsigned char *start,
        const unsigned char *end,
        const std::string &&nameFunc,
        const std::string &&kernel_mode_str,
        uint32_t sharedMemBytes) {
    if (end <= start) {
        throw std::runtime_error(std::string("invalid or empty embedded npubin range"));
    }
    return loadKernelFromData(start, static_cast<size_t>(end - start),
                              nameFunc, kernel_mode_str, sharedMemBytes);
}

static inline void * loadKernel(
        std::string filePath,
        const std::string &&nameFunc,
        uint32_t sharedMemBytes,
        const std::optional<std::string> &cubinDir = std::nullopt) {
    return loadKernel(std::move(filePath), std::move(nameFunc), inferKernelModeFromName(nameFunc),
                      sharedMemBytes, cubinDir);
}

static inline void launchKernel(
        std::function<int()> launch_call,
        const char* kernel_name) {
    at_npu::native::OpCommand::RunOpApiV2(kernel_name, launch_call);
}
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(npu);
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    void * triton_poi_fused_relu_sigmoid_tanh_0{nullptr};
};
}  // namespace



AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(3,
                           3,
                           0,
                           device_str,
                           std::move(cubin_dir),
                           true) {
    inputs_info_[0].name = "arg0_1";
    inputs_info_[1].name = "arg1_1";
    inputs_info_[2].name = "arg2_1";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}])";
    out_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}, {"type": null, "context": null, "children_spec": []}]}])";
    outputs_info_[0].name = "output0";
    outputs_info_[1].name = "output1";
    outputs_info_[2].name = "output2";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}
} // namespace torch::aot_inductor
using namespace torch::aot_inductor;

template <typename in_ptr0_type_, typename in_ptr1_type_, typename in_ptr2_type_, typename out_ptr0_type_, typename out_ptr1_type_, typename out_ptr2_type_, typename kernels_type_>
static __attribute__((noinline)) void call_triton_poi_fused_relu_sigmoid_tanh_0(
    const in_ptr0_type_& in_ptr0,
    const in_ptr1_type_& in_ptr1,
    const in_ptr2_type_& in_ptr2,
    const out_ptr0_type_& out_ptr0,
    const out_ptr1_type_& out_ptr1,
    const out_ptr2_type_& out_ptr2,
    int32_t device_idx_,
    aclrtStream stream_,
    kernels_type_& kernels_,
    const std::optional<std::string>& cubin_dir_ = std::nullopt
){
    /*
    async_compile.triton('triton_poi_fused_relu_sigmoid_tanh_0', '''
    import triton
    import triton.language as tl

    from torch._inductor.runtime import triton_helpers, triton_heuristics
    from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
    from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

    import torch
    import torch_npu
    if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
        torch_npu.npu._initialized = True
    from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
    from torch_npu._inductor.runtime import triton_helpers
    from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

    @triton_heuristics.pointwise(
        size_hints={'x0': 600}, tile_hint=TileHint.DEFAULT,
        filename=__file__,
        triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'X0BLOCK': 'i32', 'X0BLOCK_SUB': 'constexpr'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=56, cc='Ascend950PR_9579', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'mix_mode': 'aiv', 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
        inductor_meta={'grid_type': 'SequentialComboKernelGrid', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_sigmoid_tanh_0', 'mutated_arg_names': [], 'backend_hash': '<removed>', 'split_axis': [0], 'tiling_axis': [0], 'no_loop_axis': [], 'axis_names': ['x0'], 'axis_static_values': (('x0', 100),), 'low_dims': {0}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'runtime_block_arg_names': ('X0BLOCK',), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': True, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': True, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': (), 'optimize_mem': True, 'combo_grid_meta': {'num_kernels': 3, 'min_blocks': 0, 'default_config': {'XBLOCK': 'X0BLOCK'}, 'split_numels': [100, 400, 100], 'no_split': [False, False, False], 'split_block': 'X0BLOCK', 'no_x_dim_0': False, 'xnumel_0': 100, 'no_x_dim_1': False, 'xnumel_1': 400, 'no_x_dim_2': False, 'xnumel_2': 100}, 'combo_subkernel_meta': [{'axis_names': ['x0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {0}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 100}, 'persistent_reduction': False}, {'axis_names': ['x0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {0}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 400}, 'persistent_reduction': False}, {'axis_names': ['x0'], 'tiling_axis': [0], 'no_loop_axis': [], 'split_axis': [0], 'low_dims': {0}, 'split_axis_dtype': torch.float32, 'dual_reduction': False, 'npu_kernel_type': 'simt_template', 'size_hints': {'x0': 100}, 'persistent_reduction': False}]}
    )
    @triton.jit
    def triton_poi_fused_relu_sigmoid_tanh_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, X0BLOCK, X0BLOCK_SUB : tl.constexpr):
        pid = tl.program_id(0)
        num_blocks_0 = tl.cdiv(100, X0BLOCK)
        num_blocks_1 = num_blocks_0 + tl.cdiv(400, X0BLOCK)
        num_blocks_2 = num_blocks_1 + tl.cdiv(100, X0BLOCK)
        if pid < num_blocks_0:
            pid_offset = pid
            x0_numel = 100
            x0_offset = pid_offset * X0BLOCK
            base_x0= tl.arange(0, X0BLOCK_SUB)
            loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
            for loop_x0 in range(loops_x0):
                x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
                x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
                tmp0 = tl.load(in_ptr0 + (x0), x0_mask)
                tmp1 = tl.full([1], 0, tl.int32)
                tmp2 = triton_helpers.maximum(tmp1, tmp0)
                tl.store(out_ptr0 + (x0), tmp2, x0_mask)
        elif pid < num_blocks_1:
            pid_offset = pid - num_blocks_0
            x0_numel = 400
            x0_offset = pid_offset * X0BLOCK
            base_x0= tl.arange(0, X0BLOCK_SUB)
            loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
            for loop_x0 in range(loops_x0):
                x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
                x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
                tmp3 = tl.load(in_ptr1 + (x0), x0_mask)
                tmp4 = tl.sigmoid(tmp3)
                tl.store(out_ptr1 + (x0), tmp4, x0_mask)
        elif pid < num_blocks_2:
            pid_offset = pid - num_blocks_1
            x0_numel = 100
            x0_offset = pid_offset * X0BLOCK
            base_x0= tl.arange(0, X0BLOCK_SUB)
            loops_x0 = (X0BLOCK + X0BLOCK_SUB - 1) // X0BLOCK_SUB
            for loop_x0 in range(loops_x0):
                x0 = x0_offset + (loop_x0 * X0BLOCK_SUB) + base_x0
                x0_mask = x0 < min(X0BLOCK+x0_offset, x0_numel)
                tmp5 = tl.load(in_ptr2 + (x0), x0_mask)
                tmp6 = tl_math.tanh(tmp5)
                tl.store(out_ptr2 + (x0), tmp6, x0_mask)
        else:
            pass
    ''', device_str='npu')
    */
    int64_t X0BLOCK = 304;
    uint32_t grid_0 = ((100 + (X0BLOCK - 1)) / (X0BLOCK)) + ((400 + (X0BLOCK - 1)) / (X0BLOCK)) + ((100 + (X0BLOCK - 1)) / (X0BLOCK));
    uint32_t grid_1 = 1;
    uint32_t grid_2 = 1;
    if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;
    if (kernels_.triton_poi_fused_relu_sigmoid_tanh_0 == nullptr) {
        kernels_.triton_poi_fused_relu_sigmoid_tanh_0 = loadKernel("<workspace>", "triton_poi_fused_relu_sigmoid_tanh_0", "aiv", 1, cubin_dir_); 
    }
    void* var_0 = reinterpret_cast<void*>(in_ptr0.data_ptr());
    void* var_1 = reinterpret_cast<void*>(in_ptr1.data_ptr());
    void* var_2 = reinterpret_cast<void*>(in_ptr2.data_ptr());
    void* var_3 = reinterpret_cast<void*>(out_ptr0.data_ptr());
    void* var_4 = reinterpret_cast<void*>(out_ptr1.data_ptr());
    void* var_5 = reinterpret_cast<void*>(out_ptr2.data_ptr());
    int32_t var_6 = X0BLOCK;

    auto launch_call = [=]() {

        aclError ret;

        void* workspace_addr = NULL;
        void* sync_block_lock = NULL;


        struct __attribute__((packed)) {

            void* sync_block_lock __attribute__((aligned(8)));
            void* workspace_addr __attribute__((aligned(8)));
            void* var_0 __attribute__((aligned(8))); void* var_1 __attribute__((aligned(8))); void* var_2 __attribute__((aligned(8))); void* var_3 __attribute__((aligned(8))); void* var_4 __attribute__((aligned(8))); void* var_5 __attribute__((aligned(8))); int32_t var_6 __attribute__((aligned(sizeof(int32_t)))); 
            int32_t grid_0 __attribute__((aligned(4)));
            int32_t grid_1 __attribute__((aligned(4)));
            int32_t grid_2 __attribute__((aligned(4)));
            void* DTData __attribute__((aligned(8)));
        } kernel_args = {

            static_cast<void*>(sync_block_lock),
            static_cast<void*>(workspace_addr),
            static_cast<void*>(var_0), static_cast<void*>(var_1), static_cast<void*>(var_2), static_cast<void*>(var_3), static_cast<void*>(var_4), static_cast<void*>(var_5), static_cast<int32_t>(var_6), 
            static_cast<int32_t>(grid_0),
            static_cast<int32_t>(grid_1),
            static_cast<int32_t>(grid_2)
            , static_cast<void*>(nullptr)
    };

        uint32_t block_num = grid_0 * grid_1 * grid_2;
        block_num = std::min(block_num, (uint32_t)56);

        void* args = static_cast<void*>(&kernel_args);
        auto args_size = sizeof(kernel_args);
        ret = aclrtLaunchKernelWithHostArgs(kernels_.triton_poi_fused_relu_sigmoid_tanh_0, block_num, stream_, nullptr, args, args_size, nullptr, 0);

        if (ret != ACL_SUCCESS) return ret;
        return ret;
    };
    launchKernel(launch_call, "kernels_.triton_poi_fused_relu_sigmoid_tanh_0");
}


namespace torch::aot_inductor {

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

AOTI_NOINLINE static void check_input_0(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg0_1 = ConstantHandle(input_handles[0]);
    int32_t arg0_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg0_1, &arg0_1_dtype));

    int32_t arg0_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg0_1_expected_dtype != arg0_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg0_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg0_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg0_1_size = arg0_1.sizes();

    if (10 != arg0_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 10, " << "but got: " << arg0_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (10 != arg0_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 10, " << "but got: " << arg0_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg0_1_stride = arg0_1.strides();

    if (10 != arg0_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 10, " << "but got: " << arg0_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg0_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg0_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

AOTI_NOINLINE static void check_input_1(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg1_1 = ConstantHandle(input_handles[1]);
    int32_t arg1_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg1_1, &arg1_1_dtype));

    int32_t arg1_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg1_1_expected_dtype != arg1_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched dtype, "
           << "expected: " << arg1_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg1_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg1_1_size = arg1_1.sizes();

    if (20 != arg1_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched dim value at 0, "
           << "expected: 20, " << "but got: " << arg1_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (20 != arg1_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched dim value at 1, "
           << "expected: 20, " << "but got: " << arg1_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg1_1_stride = arg1_1.strides();

    if (20 != arg1_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched stride value at 0, "
           << "expected: 20, " << "but got: " << arg1_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg1_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[1]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg1_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

AOTI_NOINLINE static void check_input_2(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg2_1 = ConstantHandle(input_handles[2]);
    int32_t arg2_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg2_1, &arg2_1_dtype));

    int32_t arg2_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg2_1_expected_dtype != arg2_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dtype, "
           << "expected: " << arg2_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg2_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2_1_size = arg2_1.sizes();

    if (10 != arg2_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dim value at 0, "
           << "expected: 10, " << "but got: " << arg2_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (10 != arg2_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched dim value at 1, "
           << "expected: 10, " << "but got: " << arg2_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg2_1_stride = arg2_1.strides();

    if (10 != arg2_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched stride value at 0, "
           << "expected: 10, " << "but got: " << arg2_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg2_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[2]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg2_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

static bool _check_aoti_runtime_check_inputs_env() {
    const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
    const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
    return result;
}

AOTI_NOINLINE static void __check_inputs_outputs(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
    if (!_check_aoti_runtime_check_inputs_env()){
        return;
    }
    check_input_0(input_handles);
    check_input_1(input_handles);
    check_input_2(input_handles);
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    __check_inputs_outputs(input_handles, output_handles);
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 3);
    auto arg0_1 = std::move(inputs[0]);
    auto arg1_1 = std::move(inputs[1]);
    auto arg2_1 = std::move(inputs[2]);

    if ((reinterpret_cast<std::uintptr_t>(arg0_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 0 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg0_1_aligned;
        aoti_torch_clone_preserve_strides(arg0_1, &arg0_1_aligned);
        arg0_1 = std::move(RAIIAtenTensorHandle(arg0_1_aligned));
    }

    if ((reinterpret_cast<std::uintptr_t>(arg1_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 1 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg1_1_aligned;
        aoti_torch_clone_preserve_strides(arg1_1, &arg1_1_aligned);
        arg1_1 = std::move(RAIIAtenTensorHandle(arg1_1_aligned));
    }

    if ((reinterpret_cast<std::uintptr_t>(arg2_1.data_ptr()) & (16 -1)) != 0) {
        AOTI_TORCH_WARN("Input 2 was compiled as 16-bytes aligned, but it is not aligned at run time. Copying to an aligned tensor to guarantee correctness, but expect a performance hit.");
        AtenTensorHandle arg2_1_aligned;
        aoti_torch_clone_preserve_strides(arg2_1, &arg2_1_aligned);
        arg2_1 = std::move(RAIIAtenTensorHandle(arg2_1_aligned));
    }
    inputs.clear();
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride(arg0_1, {10L, 10L}, {10L, 1L}, "input"); }
    if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride(arg1_1, {20L, 20L}, {20L, 1L}, "input"); }
    if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride(arg2_1, {10L, 10L}, {10L, 1L}, "input"); }

    AOTINpuStreamGuard stream_guard(stream, this->device_idx_);
    static constexpr int64_t int_array_0[] = {10L, 10L};
    static constexpr int64_t int_array_1[] = {10L, 1L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_npu, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    static constexpr int64_t int_array_2[] = {20L, 20L};
    static constexpr int64_t int_array_3[] = {20L, 1L};
    AtenTensorHandle buf1_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_2, int_array_3, cached_torch_dtype_float32, cached_torch_device_type_npu, this->device_idx_, &buf1_handle));
    RAIIAtenTensorHandle buf1(buf1_handle);
    AtenTensorHandle buf2_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_npu, this->device_idx_, &buf2_handle));
    RAIIAtenTensorHandle buf2(buf2_handle);
    // Topologically Sorted Source Nodes: [relu, sigmoid, tanh], Original ATen: [aten.relu, aten.sigmoid, aten.tanh]
    // [Provenance debug handles] triton_poi_fused_relu_sigmoid_tanh_0:1
    call_triton_poi_fused_relu_sigmoid_tanh_0(arg0_1, arg1_1, arg2_1, buf0, buf1, buf2, this->device_idx_, stream, kernels, this->cubin_dir_);
    arg0_1.reset();
    arg1_1.reset();
    arg2_1.reset();
    output_handles[0] = buf0.release();
    output_handles[1] = buf1.release();
    output_handles[2] = buf2.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor




// Triton kernels are embedded as comments in <workspace>

