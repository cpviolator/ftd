#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaXEngineMatFormat, QudaPrecision>;

// The following tests gets each mat type and precision using google testing framework
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

// Specific class dervived from GoogleTest class
class XEngine : public ::testing::TestWithParam<test_t> {
  
protected:
  test_t param;
  
public:
  XEngine() : param(GetParam()) { }
};

// Forward declaration of test function to perform
double XEngineTest(test_t test_param);

// Performs the Google test and checks result
TEST_P(XEngine, verify) {
  auto deviation = XEngineTest(GetParam());
  // Test host/device deviation.
  EXPECT_EQ(deviation, 0) << "CPU and DEVICE XEngine implementations do not agree";
}

// XEngine test types to run
auto xengine_test_type_value = Values(QUDA_XENGINE_MAT_TRI, QUDA_XENGINE_MAT_HERM);

// XEngine prec types to run
auto xengine_prec_type_value = Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION);

// Test name construction
std::string gettestname(::testing::TestParamInfo<test_t> param) {
  std::string name = "XEngineTest";  
  name += std::string("_") + get_xengine_mat_format_str(::testing::get<0>(param.param)) + std::string("_") + get_prec_str(::testing::get<1>(param.param));
  return name;
}

// XEngine tests
INSTANTIATE_TEST_SUITE_P(xengine_test, XEngine, ::testing::Combine(xengine_test_type_value, xengine_prec_type_value), gettestname);
