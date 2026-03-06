#pragma once

#include <array>
#include <enum_ggp.h>
#include <util_ggp.h>
#include <ggp_internal.h>
//#include "reference_wrapper_helper.h"

namespace quda
{
  /**
     @brief Helper function for returning if a given precision is enabled
     @tparam precision The precision requested
     @return True if enabled, false if not
  */
  constexpr bool is_enabled(QudaPrecision precision) {
    switch (precision) {
    case QUDA_DOUBLE_PRECISION: return (GGP_PRECISION & 8) ? true : false;
    case QUDA_SINGLE_PRECISION: return (GGP_PRECISION & 4) ? true : false;
    case QUDA_HALF_PRECISION:   return (GGP_PRECISION & 2) ? true : false;
    case QUDA_QUARTER_PRECISION:  return (GGP_PRECISION & 1) ? true : false;
    default: return false;
    }
  }

  /**
     @brief This instantiate function is used to instantiate the precisions
     @param[in] F DataField The data field
     @param[in,out] args Any additional arguments required for the computation at hand
  */
  template <template <typename, int> class Apply, typename F,
            typename... Args>
  constexpr void instantiate(F &field, Args &&...args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      if constexpr (is_enabled(QUDA_DOUBLE_PRECISION))
        instantiate<Apply, double>(field, args...);
      else
        errorQuda("GGP_PRECISION=%d does not enable double precision", GGP_PRECISION);
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
        instantiate<Apply, float>(field, args...);
      else
        errorQuda("GGP_PRECISION=%d does not enable single precision", GGP_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }
  
  /**
     @brief This instantiate2 function is used to instantiate the
     precisions, with double precision always enabled.  This is a
     temporary addition until we fuse this with the original function
     above when we enforce C++17
     @param[in] F DataField the data field object
     @param[in,out] args Any additional arguments required for the computation at hand
  */
  template <template <typename, int> class Apply, typename F,
            typename... Args>
  constexpr void instantiate2(F &field, Args &&...args)
  {
    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      instantiate<Apply, double>(field, args...);
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION))
        instantiate<Apply, float>(field, args...);
      else
        errorQuda("GGP_PRECISION=%d does not enable single precision", GGP_PRECISION);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }

  /**
     @brief The instantiatePrecision function is used to instantiate
     the precision.  Note unlike the "instantiate" functions above,
     this helper always instantiates double precision regardless of
     the GGP_PRECISION value: this enables its use for copy interface
     routines which should always enable double precision support.

     @param[in] field DataField we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename> class Apply, typename F, typename... Args>
  constexpr void instantiatePrecision(F &field, Args &&... args)
  {
    if (!is_enabled(field.Precision()) && field.Precision() != QUDA_DOUBLE_PRECISION)
      errorQuda("GGP_PRECISION=%d does not enable %d precision", GGP_PRECISION, field.Precision());

    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      Apply<double>(field, args...); // always instantiate double precision
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) Apply<float>(field, args...);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) Apply<short>(field, args...);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) Apply<int8_t>(field, args...);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }

  /**
     @brief The instantiatePrecision function is used to instantiate
     the precision.  Note unlike the "instantiate" functions above,
     this helper always instantiates double precision regardless of
     the GGP_PRECISION value: this enables its use for copy interface
     routines which should always enable double precision support.

     @param[in] QudaPrecision prec The precision we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename> class Apply, typename... Args>
  constexpr void instantiatePrecisionTEST(QudaPrecision prec, Args &&... args)
  {
    if (!is_enabled(prec) && prec != QUDA_DOUBLE_PRECISION)
      errorQuda("GGP_PRECISION=%d does not enable %d precision", GGP_PRECISION, prec);

    if (prec == QUDA_DOUBLE_PRECISION) {
      Apply<double>(prec, args...); // always instantiate double precision
    } else if (prec == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) Apply<float>(prec, args...);
    } else if (prec == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) Apply<short>(prec, args...);
    } else if (prec == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) Apply<int8_t>(prec, args...);
    } else {
      errorQuda("Unsupported precision %d\n", prec);
    }
  }

  
  /**
     @brief The instantiatePrecision2 function is used to instantiate
     the precision for a class that accepts 2 typename arguments, with
     the first typename corresponding to the precision being
     instantiated at hand.  This is useful for copy routines, where we
     need to instantiate a second, e.g., destination, precision after
     already instantiating the first, e.g., source, precision.
     Similar to the "instantiatePrecision" function above, this helper
     always instantiates double precision regardless of the
     GGP_PRECISION value: this enables its use for copy interface
     routines which should always enable double precision support.

     @param[in] field LatticeField we wish to instantiate
     @param[in,out] args Any additional arguments required for the
     computation at hand
  */
  template <template <typename, typename> class Apply, typename T, typename F, typename... Args>
  constexpr void instantiatePrecision2(F &field, Args &&... args)
  {
    if (!is_enabled(field.Precision()) && field.Precision() != QUDA_DOUBLE_PRECISION)
      errorQuda("GGP_PRECISION=%d does not enable %d precision", GGP_PRECISION, field.Precision());

    if (field.Precision() == QUDA_DOUBLE_PRECISION) {
      Apply<double, T>(field, args...); // always instantiate double precision
    } else if (field.Precision() == QUDA_SINGLE_PRECISION) {
      if constexpr (is_enabled(QUDA_SINGLE_PRECISION)) Apply<float, T>(field, args...);
    } else if (field.Precision() == QUDA_HALF_PRECISION) {
      if constexpr (is_enabled(QUDA_HALF_PRECISION)) Apply<short, T>(field, args...);
    } else if (field.Precision() == QUDA_QUARTER_PRECISION) {
      if constexpr (is_enabled(QUDA_QUARTER_PRECISION)) Apply<int8_t, T>(field, args...);
    } else {
      errorQuda("Unsupported precision %d\n", field.Precision());
    }
  }


#ifdef GPU_DISTANCE_PRECONDITIONING
  constexpr bool is_enabled_distance_precondition() { return true; }
#else
  constexpr bool is_enabled_distance_precondition() { return false; }
#endif

} // namespace quda
