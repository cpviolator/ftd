#include <gtest/gtest.h>

using test_t = ::testing::tuple<QudaPrecision>;

using ::testing::TestWithParam;
using ::testing::Values;

// Forward declarations of verification test functions
double VerificationTest_Small();
double VerificationTest_Medium();
double VerificationTest_Large();
double VerificationTest_LargeK();
double VerificationTest_TCC();

class CutlassVerifySmall : public ::testing::TestWithParam<test_t> {};
class CutlassVerifyMedium : public ::testing::TestWithParam<test_t> {};
class CutlassVerifyLarge : public ::testing::TestWithParam<test_t> {};
class CutlassVerifyLargeK : public ::testing::TestWithParam<test_t> {};
class CutlassVerifyTCC : public ::testing::TestWithParam<test_t> {};

TEST_P(CutlassVerifySmall, verify) {
  double err = VerificationTest_Small();
  EXPECT_LT(err, 0.5) << "Small verification (N=32, K=8) failed";
}

TEST_P(CutlassVerifyMedium, verify) {
  double err = VerificationTest_Medium();
  EXPECT_LT(err, 0.5) << "Medium verification (N=64, K=8) failed";
}

TEST_P(CutlassVerifyLarge, verify) {
  double err = VerificationTest_Large();
  EXPECT_LT(err, 0.5) << "Large verification (N=128, K=8) failed";
}

TEST_P(CutlassVerifyLargeK, verify) {
  double err = VerificationTest_LargeK();
  EXPECT_LT(err, 0.5) << "Large-K verification (N=256, K=32) failed";
}

TEST_P(CutlassVerifyTCC, verify) {
#ifndef GGP_TCC_ENABLED
  GTEST_SKIP() << "TCC not enabled (build with -DGGP_TCC=ON)";
#endif
  double err = VerificationTest_TCC();
  EXPECT_LT(err, 0.5) << "TCC-aligned verification (N=32, K=128) failed";
}

auto benchmark_prec = Values(QUDA_SINGLE_PRECISION);
INSTANTIATE_TEST_SUITE_P(cutlass_benchmark, CutlassVerifySmall, benchmark_prec);
INSTANTIATE_TEST_SUITE_P(cutlass_benchmark, CutlassVerifyMedium, benchmark_prec);
INSTANTIATE_TEST_SUITE_P(cutlass_benchmark, CutlassVerifyLarge, benchmark_prec);
INSTANTIATE_TEST_SUITE_P(cutlass_benchmark, CutlassVerifyLargeK, benchmark_prec);
INSTANTIATE_TEST_SUITE_P(cutlass_benchmark, CutlassVerifyTCC, benchmark_prec);
