#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaPrecision>;

// The following tests gets each mat type and precision using google testing framework
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

// Specific class dervived from GoogleTest class
class Beamformer : public ::testing::TestWithParam<test_t> {
  
protected:
  test_t param;
  
public:
  Beamformer() : param(GetParam()) { }
};

// Forward declaration of test function to perform
double BeamformerTest(test_t test_param);

// Performs the Google test and checks result
TEST_P(Beamformer, verify) {
  auto deviation = BeamformerTest(GetParam());
  // Test host/device deviation.
  EXPECT_EQ(deviation, 0) << "CPU and DEVICE Beamformer implementations do not agree";
}

// Beamformer prec types to run
//auto beamformer_prec_type_value = Values(QUDA_DOUBLE_PRECISION, QUDA_SINGLE_PRECISION);
auto beamformer_prec_type_value = Values(QUDA_SINGLE_PRECISION);

// Test name construction
std::string gettestname(::testing::TestParamInfo<test_t> param) {
  std::string name = "BeamformerTest";  
  name += std::string("_") + get_prec_str(::testing::get<0>(param.param));
  return name;
}

// Beamformer tests
INSTANTIATE_TEST_SUITE_P(beamformer_test, Beamformer, ::testing::Combine(beamformer_prec_type_value), gettestname);
