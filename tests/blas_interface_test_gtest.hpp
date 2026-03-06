#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaBLASType, QudaBLASDataType, QudaBLASEngine>;

class BLASTest : public ::testing::TestWithParam<test_t>
{
protected:
  test_t param;

public:
  BLASTest() : param(GetParam()) { }
};

bool skip_test(test_t param)
{
  auto test_type = ::testing::get<0>(param);
  auto data_type = ::testing::get<1>(param);
  auto engine = ::testing::get<2>(param);

  // LU_INV only supports complex types (C, Z)
  if (test_type == QUDA_BLAS_LU_INV
      && (data_type == QUDA_BLAS_DATATYPE_D || data_type == QUDA_BLAS_DATATYPE_S))
    return true;

  // CUTLASS engine only supports HERK pattern (not general GEMM or LU_INV)
  if (engine == QUDA_BLAS_ENGINE_CUTLASS)
    return true;

  return false;
}

// For googletest, names must be non-empty, unique, and may only contain ASCII
// alphanumeric characters or underscore.
const char *data_type_str[] = {
  "realSingle",
  "realDouble",
  "complexSingle",
  "complexDouble",
};

const char *test_type_str[] = {
  "GEMM",
  "LUInvert",
};

const char *engine_type_str[] = {
  "CUBLAS",   // QUDA_BLAS_ENGINE_CUBLAS = 0
  "CUTLASS",  // QUDA_BLAS_ENGINE_CUTLASS = 1
};

// Helper functions to construct the test name
std::string getBLASDataName(testing::TestParamInfo<test_t> param)
{
  auto data_type = ::testing::get<1>(param.param);
  std::string str(data_type_str[data_type]);
  return str;
}

std::string getBLASTestName(testing::TestParamInfo<test_t> param)
{
  auto test_type = ::testing::get<0>(param.param);
  std::string str(test_type_str[test_type]);
  return str;
}

std::string getBLASEngineName(testing::TestParamInfo<test_t> param)
{
  auto engine_type = ::testing::get<2>(param.param);
  std::string str(engine_type_str[engine_type]);
  return str;
}


// The following tests gets each BLAS type and precision using google testing framework
//using ::testing::Bool;
using ::testing::Combine;
//using ::testing::Range;
using ::testing::TestWithParam;
using ::testing::Values;

double gemm_test(test_t test_param);
double lu_inv_test(test_t test_param);

// Sets up the Google test
TEST_P(BLASTest, verify)
{
  if (skip_test(GetParam())) GTEST_SKIP();

  auto param = GetParam();
  auto test_type = ::testing::get<0>(param);
  auto data_type = ::testing::get<1>(param);
  auto engine_type = ::testing::get<2>(param);
  switch (test_type) {
  case QUDA_BLAS_GEMM: {
    auto deviation_gemm = gemm_test(param);
    decltype(deviation_gemm) tol_gemm = 0.0; // initialize to suppress warning
    switch (data_type) {
    case QUDA_BLAS_DATATYPE_S:
    case QUDA_BLAS_DATATYPE_C: tol_gemm = 10 * std::numeric_limits<float>::epsilon(); break;
    case QUDA_BLAS_DATATYPE_D:
    case QUDA_BLAS_DATATYPE_Z: tol_gemm = 10 * std::numeric_limits<double>::epsilon(); break;
    default: errorQuda("Unexpected BLAS data type %d", data_type);
    }
    EXPECT_LE(deviation_gemm, tol_gemm) << "CPU and CUDA GEMM implementations do not agree";
    break;
  }
  case QUDA_BLAS_LU_INV: {
    auto deviation_lu_inv = lu_inv_test(param);
    decltype(deviation_lu_inv) tol_lu_inv = 0.0; // initialize to suppress warning
    // We allow a factor of 5000 (500x more than the gemm tolerance factor)
    // due to variations in algorithmic implementation, order of arithmetic
    // operations, and possible near singular eigenvalues or degeneracies.
    switch (data_type) {
    case QUDA_BLAS_DATATYPE_C: tol_lu_inv = 5000 * std::numeric_limits<float>::epsilon(); break;
    case QUDA_BLAS_DATATYPE_Z: tol_lu_inv = 5000 * std::numeric_limits<double>::epsilon(); break;
    case QUDA_BLAS_DATATYPE_S:
    case QUDA_BLAS_DATATYPE_D:
    default: errorQuda("Unexpected BLAS data type %d", data_type);
    }
    EXPECT_LE(deviation_lu_inv, tol_lu_inv) << "CPU and CUDA LU Inversion implementations do not agree";
    break;
  }
  default: errorQuda("Unexpected BLAS test type %d", test_type);
  }
}

// BLAS test type
auto blas_test_type_value = Values(QUDA_BLAS_GEMM, QUDA_BLAS_LU_INV);

// BLAS data type
auto blas_data_type_value
  = Values(QUDA_BLAS_DATATYPE_Z, QUDA_BLAS_DATATYPE_C, QUDA_BLAS_DATATYPE_D, QUDA_BLAS_DATATYPE_S);

// BLAS engine type
auto blas_engine_type_value
  = Values(QUDA_BLAS_ENGINE_CUTLASS, QUDA_BLAS_ENGINE_CUBLAS);

std::string gettestname(::testing::TestParamInfo<test_t> param)
{
  std::string name = "BLASTest";
  name += std::string("_") + getBLASTestName(param) + std::string("_") + getBLASDataName(param) + std::string("_") + getBLASEngineName(param);
  return name;
}

// BLAS tests
INSTANTIATE_TEST_SUITE_P(BLASTest, BLASTest, ::testing::Combine(blas_test_type_value, blas_data_type_value, blas_engine_type_value), gettestname);
