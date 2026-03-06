#pragma once

#include <CLI11.hpp>
#include <array>
#include <ggp.h>

// for compatibility while porting - remove later
extern void usage(char **);

class GGPApp : public CLI::App
{

public:
  GGPApp(std::string app_description = "", std::string app_name = "") : CLI::App(app_description, app_name) {};

  virtual ~GGPApp() {};

};

std::shared_ptr<GGPApp> make_app(std::string app_description = "GGP internal test", std::string app_name = "");

void add_comms_option_group(std::shared_ptr<GGPApp> ggp_app);
void add_testing_option_group(std::shared_ptr<GGPApp> ggp_app);
//void add_xengine_option_group(std::shared_ptr<GGPApp> ggp_app);

template <typename T> std::string inline get_string(CLI::TransformPairs<T> &map, T val)
{
  auto it
    = std::find_if(map.begin(), map.end(), [&val](const decltype(map.back()) &p) -> bool { return p.second == val; });
  return it->first;
}

// General device/topology/testing options
//----------------------------------------
extern int device_ordinal;
extern int rank_order;
extern bool native_blas_lapack;

extern bool verify_results;
extern bool enable_testing;

extern unsigned long long n_elems;

extern std::array<int, 4> gridsize_from_cmdline;
extern std::array<int, 4> dim_partitioned;
extern std::array<int, 4> grid_partition;

extern QudaPrecision prec;
extern QudaPrecision compute_prec;
extern QudaPrecision output_prec;
extern QudaVerbosity verbosity;
extern std::array<int, 4> dim;
extern int &xdim;
extern int &ydim;
extern int &zdim;
extern int &tdim;
//----------------------------------------

// XEngine options
//----------------------
/*
extern int packet_size;
extern int XE_n_packets_per_block;
extern int XE_n_antennae;
extern int XE_n_channels_per_packet;
extern int XE_n_base;
extern int XE_n_polarizations;
extern QudaXEngineMatFormat XE_mat_format;
*/
