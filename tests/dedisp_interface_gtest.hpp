#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaPrecision, QudaPrecision>;

// The following tests gets each mat type and precision using google testing framework
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

// Specific class dervived from GoogleTest class
class Dedisp : public ::testing::TestWithParam<test_t> {
  
protected:
  test_t param;
  
public:
  Dedisp() : param(GetParam()) { }
};

// Forward declaration of test function to perform
double DedispTest(test_t test_param);

// Performs the Google test and checks result
TEST_P(Dedisp, verify) {
  auto deviation = DedispTest(GetParam());
  // Test host/device deviation.
  EXPECT_EQ(deviation, 0) << "CPU and DEVICE Dedisp implementations do not agree";
}

// Dedisp prec types to run
auto dedisp_compute_prec_type_value = Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION);
auto dedisp_storage_prec_type_value = Values(QUDA_SINGLE_PRECISION, QUDA_DOUBLE_PRECISION);

// Test name construction
std::string gettestname(::testing::TestParamInfo<test_t> param) {
  std::string name = "DedispTest";  
  name += std::string("_") + get_prec_str(::testing::get<0>(param.param));
  name += std::string("_") + get_prec_str(::testing::get<1>(param.param));
  return name;
}

// Dedisp tests
INSTANTIATE_TEST_SUITE_P(dedisp_test, Dedisp, ::testing::Combine(dedisp_compute_prec_type_value, dedisp_storage_prec_type_value), gettestname);
